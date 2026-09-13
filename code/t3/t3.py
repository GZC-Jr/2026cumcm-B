#!/usr/bin/env python3
"""问题三中心点加8个外围点滚动路径顺路补测版V2，外围半径939米，Python 3.9+，依赖 numpy scipy。
安装：python -m pip install numpy scipy
官方模拟器：python t3.py --robot-id 参赛队号
本地自测：python t3.py --self-test 10
默认布局为原点加半径939米的正八边形，并保留V1的同点批量频道检测。为已有目标选择下一检测点时，优先考察当前点至滚动路线下一必经点的线段和该必经点，在交会信息损失可接受时采用零绕行补测点。
只适用于全向源。统计假设为位置均匀先验、固定接收半径均匀先验、不同测点高斯测角近似。
各频道存在性为独立伯努利近似(默认0.65)，未建模10~16总数的联合概率约束。
这些概率只用于选址；完成条件和清除认证始终采用几何条件。
不是官方模拟器，不读取官方隐藏数据。--self-test 仅为本地合成案例。
"""
import numpy as np
from scipy.optimize import minimize, linprog
import argparse
import csv
import hashlib
import json
import math
import random
import time
import uuid
from pathlib import Path
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / 'outputs' / 't3'
DEFAULT_LOCAL_TEST_OUTPUT = OUTPUT_ROOT / 'local_tests'
DEFAULT_RUN_OUTPUT = OUTPUT_ROOT / 'runs'

DELTA = math.radians(1.006)  # ±1度加两位小数取整裕量
N_CIRCLE = 96
RECEIVE_RADIUS_M = 1000.0
# Candidate points keep a small strict margin from the 1000 m boundary.  The
# squared margin is used by the triangle-vertex certificate; using one shared
# value prevents bayes_select and route_aligned_measurement from disagreeing.
RECEIVE_MARGIN_M = 1e-6
RECEIVE_MARGIN_M2 = (RECEIVE_RADIUS_M ** 2 -
                     (RECEIVE_RADIUS_M - RECEIVE_MARGIN_M) ** 2)

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def clip(poly, a, b):
    """裁剪 a·x<=b，返回凸多边形；边界使用微小外放容差。"""
    if not poly:
        return []
    b += 1e-8
    out = []
    prev = poly[-1]
    fp = a[0]*prev[0]+a[1]*prev[1]-b
    for cur in poly:
        fc = a[0]*cur[0]+a[1]*cur[1]-b
        if (fp <= 0) != (fc <= 0):
            t = fp/(fp-fc)
            out.append((prev[0]+t*(cur[0]-prev[0]), prev[1]+t*(cur[1]-prev[1])))
        if fc <= 0:
            out.append(cur)
        prev, fp = cur, fc
    return out

def disk_clip(poly, center, radius):
    # 圆的切线半平面：外接近似，不能用内接多边形排除真实目标。
    for k in range(N_CIRCLE):
        t = 2*math.pi*k/N_CIRCLE
        a = (math.cos(t), math.sin(t))
        poly = clip(poly, a, radius+a[0]*center[0]+a[1]*center[1])
    return poly

def initial_poly():
    return disk_clip([(-1800,-1800),(1800,-1800),(1800,1800),(-1800,1800)], (0,0),1800)

def normalize_points(points):
    """Return a finite (n, 2) point array, including an empty array."""
    values=np.asarray(points,dtype=float)
    if values.size==0:return np.empty((0,2),dtype=float)
    if values.ndim==1 and values.size==2:values=values.reshape(1,2)
    if values.ndim!=2 or values.shape[1]!=2:
        raise ValueError('点集合必须是形状为(n,2)的坐标数组')
    if not np.all(np.isfinite(values)):raise ValueError('点集合包含非有限坐标')
    return values

def reception_upper_bound_m2(triangles,candidate,stations=()):
    """Return the worst triangle certificate value in square metres.

    For each triangle, the candidate is accepted if either its 1000 m disk or
    one of the previously received station disks covers every vertex.  The
    maximum of the corresponding convex-quadratic vertex bounds is a
    conservative continuous-region test.  An empty triangle mesh is treated
    as uncertifiable rather than silently accepted.
    """
    triangles=np.asarray(triangles,dtype=float)
    if triangles.ndim!=3 or triangles.shape[1:]!=(3,2) or len(triangles)==0:
        return math.inf
    candidate=np.asarray(candidate,dtype=float)
    if candidate.shape!=(2,) or not np.all(np.isfinite(candidate)):return math.inf
    stations=normalize_points(stations)
    d2=np.sum((triangles-candidate)**2,axis=2)
    upper=np.max(d2-RECEIVE_RADIUS_M**2,axis=1)
    if len(stations):
        station_r2=np.sum((triangles[:,:,None,:]-stations[None,None,:,:])**2,axis=3)
        station_upper=np.min(np.max(d2[:,:,None]-station_r2,axis=1),axis=1)
        upper=np.minimum(upper,station_upper)
    return float(np.max(upper))

def reception_certified(upper_m2):
    """Apply the shared strict numerical margin to a certificate value."""
    return math.isfinite(float(upper_m2)) and float(upper_m2)<=-RECEIVE_MARGIN_M2

def update_poly(poly, q, angle):
    t = math.radians(angle)
    for a in [(math.sin(t-DELTA),-math.cos(t-DELTA)),
              (-math.sin(t+DELTA),math.cos(t+DELTA))]:
        poly = clip(poly,a,a[0]*q[0]+a[1]*q[1])
    poly = disk_clip(poly,q,1500)
    if not poly:
        raise RuntimeError('定位约束为空：检查角度、题目类型或误差界；停止，避免误判完成。')
    return poly

def contains(poly, p):
    return all((b[0]-a[0])*(p[1]-a[1])-(b[1]-a[1])*(p[0]-a[0]) >= -1e-6
               for a,b in zip(poly,poly[1:]+poly[:1]))

def circum(a,b,c):
    # 相对坐标计算，减少消减误差。
    ux,uy=b[0]-a[0],b[1]-a[1]
    vx,vy=c[0]-a[0],c[1]-a[1]
    det=2*(ux*vy-uy*vx)
    if abs(det)<1e-12:
        return None
    u2,v2=ux*ux+uy*uy,vx*vx+vy*vy
    return (a[0]+(u2*vy-v2*uy)/det,a[1]+(ux*v2-vx*u2)/det)

def mec(poly):
    """枚举2/3个支撑顶点，小型定位多边形适用；最终全顶点认证。"""
    if len(poly)==0:
        # An empty feasible region has no finite covering circle.  Returning a
        # structured non-certificate keeps diagnostic callers from failing in
        # a reduction operation; schedulers must still treat radius=inf as
        # uncertifiable.
        return (0.0,0.0),math.inf
    if len(poly)==1:
        return poly[0],0.0
    center=(sum(p[0] for p in poly)/len(poly),sum(p[1] for p in poly)/len(poly))
    best=max(dist(center,p) for p in poly)
    def consider(c,r):
        nonlocal center,best
        if r<best and all(dist(c,p)<=r+1e-7 for p in poly):
            center,best=c,max(dist(c,p) for p in poly)
    for i,a in enumerate(poly):
        for j in range(i):
            b=poly[j]
            c=((a[0]+b[0])/2,(a[1]+b[1])/2)
            consider(c,dist(a,b)/2)
            for k in range(j):
                c=circum(a,b,poly[k])
                if c is not None:
                    consider(c,dist(c,a))
    return center,max(dist(center,p) for p in poly)

def polygon_diameter(poly):
    """问题一证书：凸多边形直径由顶点对取得；顶点少，直接准确枚举。"""
    if len(poly)<2:return 0.
    return max(dist(poly[i],poly[j]) for i in range(len(poly)) for j in range(i))

def problem1_geometry_certificate(poly,center=None,radius=None):
    """记录 MEC 与 Jung 上界；只作证书，不参与任何调度决策。"""
    if center is None or radius is None:center,radius=mec(poly)
    diameter=polygon_diameter(poly);jung_upper=diameter/math.sqrt(3)
    tol=1e-7*max(1.,diameter)
    ratio=radius/jung_upper if jung_upper>tol else (0. if radius<=tol else math.inf)
    radius_to_diameter=radius/diameter if diameter>tol else (0. if radius<=tol else math.inf)
    return {'vertex_count':len(poly),'diameter_m':diameter,'mec_center':list(center),
            'mec_radius_m':radius,'jung_upper_m':jung_upper,'jung_ratio':ratio,
            'radius_to_diameter':radius_to_diameter,
            'jung_certified':radius<=jung_upper+tol,'numeric_tolerance_m':tol}

# 可通过命令行调整。权重是选址偏好，不是目标位置的概率密度。
SCHEDULE_CONFIG = {'search_gain':0.2,'lookahead':1.0,'cell_size':100.}

def mesh(poly, depth):
    v=normalize_points(poly)
    if len(v)<3:return np.empty((0,3,2),dtype=float),np.empty(0,dtype=float)
    center=v.mean(axis=0)
    tri=np.array([[center,a,b] for a,b in zip(v,np.roll(v,-1,axis=0))])
    for _ in range(depth):
        if not len(tri):break
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        tri=np.concatenate([np.stack(x,axis=1) for x in
            [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
    if not len(tri):return np.empty((0,3,2),dtype=float),np.empty(0,dtype=float)
    a,b=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    area=np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])/2
    return tri[area>1e-10],area[area>1e-10]

BAYES_CONFIG={'noise_deg':1/math.sqrt(3),'quad_depth':3,'prior_exists':0.65}

# 问题二仅作 V2 主目标近乎同值时的决胜项；默认参数不改动任何 V2 配置。
Q12_LINK_CONFIG={'eps_w':0.2,'p_w':2.0,'tau':0.01,'min_dop_improvement':0.02,
                 'max_primary_loss_ratio':0.005,'primary_tie_rel_tol':1e-6,
                 'sin_floor':0.02,'collinear_sin_tol':1e-8,
                 'max_collinear_weight_fraction':0.05,'enable_dop_override':True}

def point_segment_distances(points,a,b):
    """批量计算点到线段 ab 的欧氏距离，兼容退化边。"""
    edge=b-a;length2=float(edge@edge)
    if length2<=1e-24:return np.linalg.norm(points-a,axis=1)
    t=np.clip((points-a)@edge/length2,0.,1.)
    return np.linalg.norm(points-(a+t[:,None]*edge),axis=1)

def depth_weighted_dop(poly,station1,candidate,quad_depth=2):
    """以三角形面积求积近似连续区域深度加权 DOP。"""
    # This helper is also used by diagnostic callers, so malformed points must
    # be reported as an unusable (infinite) score rather than bubbling up a
    # numpy broadcasting/shape exception.  The scheduler treats ``inf`` as a
    # tie-break failure and keeps the hard geometric certificate in charge.
    try:
        station1_value=np.asarray(station1,dtype=float)
        candidate_value=np.asarray(candidate,dtype=float)
    except (TypeError,ValueError):
        return math.inf
    if (station1_value.shape!=(2,) or candidate_value.shape!=(2,) or
            not np.all(np.isfinite(station1_value)) or
            not np.all(np.isfinite(candidate_value))):
        return math.inf
    try:
        depth_value=float(quad_depth)
        if (not math.isfinite(depth_value) or depth_value<0 or
                depth_value!=math.floor(depth_value) or depth_value>10):
            return math.inf
        vertices=normalize_points(poly)
        if len(vertices)<3:
            return math.inf
        triangles,areas=mesh(vertices,int(depth_value))
    except (TypeError,ValueError,IndexError,OverflowError):
        return math.inf
    points=triangles.mean(axis=1)
    if (not len(points) or not np.all(np.isfinite(points)) or
            not np.all(np.isfinite(areas)) or np.any(areas<=0)):
        return math.inf
    depths=np.full(len(points),np.inf)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        depths=np.minimum(depths,point_segment_distances(points,a,b))
    d_star=float(np.max(depths))
    if not math.isfinite(d_star) or d_star<=1e-12:return math.inf
    cfg=Q12_LINK_CONFIG
    weights=areas*(cfg['eps_w']+(1-cfg['eps_w'])*(np.maximum(depths,0.)/d_star)**cfg['p_w'])
    if not np.all(np.isfinite(weights)) or np.any(weights<0):
        return math.inf
    u=points-station1_value;v=points-candidate_value
    r1=np.linalg.norm(u,axis=1);r2=np.linalg.norm(v,axis=1)
    denom=r1*r2
    cross=np.abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])
    sin_alpha=np.divide(cross,denom,out=np.zeros_like(cross),where=denom>1e-12)
    bad=(denom<=1e-12)|(~np.isfinite(sin_alpha))|(sin_alpha<=cfg['collinear_sin_tol'])
    total=float(weights.sum())
    if (not math.isfinite(total) or total<=1e-300 or
            float(weights[bad].sum())/total>cfg['max_collinear_weight_fraction']):return math.inf
    dop=np.sqrt(r1*r1+r2*r2)/np.maximum(np.abs(sin_alpha),cfg['sin_floor'])
    value=float(weights@dop/total)
    return value if math.isfinite(value) else math.inf

DEFAULT_PRIOR_COVARIANCE = np.eye(2, dtype=float) * (1800.0 ** 2 / 4.0)


def _project_covariance(covariance, floor=1e-4):
    """Return a finite positive-definite 2-D covariance, or ``None``.

    The planning posterior is only an approximation, so a singular matrix is
    not a reason to abort the whole run.  Eigenvalue flooring is deliberately
    kept in one helper so every fallback uses the same numerical convention.
    """
    try:
        value = np.asarray(covariance, dtype=float)
    except (TypeError, ValueError):
        return None
    if value.shape != (2, 2) or not np.all(np.isfinite(value)):
        return None
    try:
        value = (value + value.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(value)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(eigenvalues)):
        return None
    return (eigenvectors * np.maximum(eigenvalues, float(floor))) @ eigenvectors.T


def _finite_point(point, default=None):
    try:
        value = np.asarray(point, dtype=float)
    except (TypeError, ValueError):
        value = np.empty(0, dtype=float)
    if value.shape == (2,) and np.all(np.isfinite(value)):
        return value
    if default is not None:
        try:
            fallback = np.asarray(default, dtype=float)
            if fallback.shape == (2,) and np.all(np.isfinite(fallback)):
                return fallback
        except (TypeError, ValueError):
            pass
    return np.zeros(2, dtype=float)


def _fallback_moments(poly=None, fallback_point=None, fallback_covariance=None):
    """Build finite moments when a quadrature mesh is empty or underflows."""
    try:
        vertices = normalize_points([] if poly is None else poly)
    except (TypeError, ValueError):
        vertices = np.empty((0, 2), dtype=float)
    if len(vertices):
        try:
            return gaussian_moments(vertices, np.ones(len(vertices), dtype=float))
        except (TypeError, ValueError, np.linalg.LinAlgError):
            pass
    mean = _finite_point(fallback_point)
    covariance = _project_covariance(fallback_covariance)
    if covariance is None:
        covariance = DEFAULT_PRIOR_COVARIANCE.copy()
    return mean, covariance


def _posterior_record(mu, covariance, note, quadrature_points, noise_std_deg=None):
    covariance = _project_covariance(covariance)
    if covariance is None:
        covariance = DEFAULT_PRIOR_COVARIANCE.copy()
    sign, logdet = np.linalg.slogdet(covariance)
    entropy = (float(math.log(2 * math.pi * math.e) + 0.5 * logdet)
               if sign > 0 and math.isfinite(float(logdet)) else None)
    record = {'mean': np.asarray(mu, dtype=float).tolist(),
              'covariance': covariance.tolist(), 'note': note,
              'quadrature_points': int(quadrature_points),
              'gaussian_entropy_nats': entropy}
    if noise_std_deg is not None:
        record['noise_std_deg'] = float(noise_std_deg)
    return record


def gaussian_moments(points,weights):
    """Return normalized weighted moments with explicit input validation."""
    points = normalize_points(points)
    try:
        w = np.asarray(weights, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('后验权重格式无效') from exc
    if w.ndim != 1 or len(w) != len(points) or not np.all(np.isfinite(w)):
        raise ValueError('后验权重格式无效')
    if np.any(w < 0):
        raise ValueError('后验权重不能为负')
    total = float(w.sum())
    if not math.isfinite(total) or total <= 1e-300:
        raise ValueError('后验积分权重退化')
    w = w / total
    mu = w @ points
    if not np.all(np.isfinite(mu)):
        raise ValueError('后验均值非有限')
    diff = points - mu
    covariance = (diff * w[:, None]).T @ diff
    covariance = _project_covariance(covariance)
    if covariance is None:
        raise ValueError('后验协方差非有限')
    return mu, covariance

def radius_interval(points,positives,negatives):
    points=normalize_points(points)
    positives=normalize_points(positives)
    negatives=normalize_points(negatives)
    lower=np.full(len(points),1000.);upper=np.full(len(points),1500.)
    for q in positives:lower=np.maximum(lower,np.linalg.norm(points-q,axis=1))
    for q in negatives:upper=np.minimum(upper,np.linalg.norm(points-q,axis=1))
    return lower,upper

def update_gaussian(target,q,angle):
    """受几何支持集约束的假定密度滤波(ADF)。
    p_new(G) ∝ N(G;mu,Sigma)*测角似然*接收半径条件似然*I(G在可行域)。
    对非高斯后验积分求均值/协方差，再用二维高斯近似。
    首次更新采用圆域内均匀先验；R~U[1000,1500]且固定，跨观测共享。
    """
    if not isinstance(target,dict):
        raise ValueError('目标后验必须是字典')
    poly=target.get('poly',[])
    try:
        triangles,areas=mesh(poly,BAYES_CONFIG['quad_depth'])
    except (TypeError,ValueError,IndexError):
        triangles=np.empty((0,3,2),dtype=float);areas=np.empty(0,dtype=float)
    fallback_mean,fallback_cov=_fallback_moments(poly,_finite_point(q))
    prior_mean=_finite_point(target.get('mean'),fallback_mean)
    prior_cov=_project_covariance(target.get('covariance'))
    if prior_cov is None:prior_cov=fallback_cov.copy()
    try:
        q_value=np.asarray(q,dtype=float)
        if q_value.shape!=(2,) or not np.all(np.isfinite(q_value)):
            raise ValueError
    except (TypeError,ValueError):
        q_value=fallback_mean.copy()
    history_note=''
    try:
        positives=normalize_points(target.get('stations',[]))
        negatives=normalize_points(target.get('negatives',[]))
    except (TypeError,ValueError):
        positives=np.empty((0,2),dtype=float);negatives=np.empty((0,2),dtype=float)
        history_note='invalid_history_fallback'
    try:
        angle_value=float(angle)
        if not math.isfinite(angle_value):raise ValueError
    except (TypeError,ValueError):
        angle_value=0.;history_note='invalid_angle_fallback'
    try:
        noise_std=float(BAYES_CONFIG['noise_deg'])
        if not math.isfinite(noise_std) or noise_std<=0:raise ValueError
    except (TypeError,ValueError):
        noise_std=1/math.sqrt(3);history_note='invalid_noise_fallback'
    if not len(triangles):
        target['mean']=prior_mean;target['covariance']=prior_cov
        return _posterior_record(prior_mean,prior_cov,
                                 'degenerate_mesh_fallback'+(
                                     ':'+history_note if history_note else ''),0,noise_std)
    points=triangles.mean(axis=1)
    lower,upper=radius_interval(points,positives,negatives)
    new_width=np.maximum(upper-lower,0.)
    delta=points-q_value;distance=np.linalg.norm(delta,axis=1)
    residual=(np.arctan2(delta[:,1],delta[:,0])-math.radians(angle_value)+math.pi)%(2*math.pi)-math.pi
    variance=math.radians(noise_std)**2
    logw=np.log(np.maximum(areas,np.finfo(float).tiny))-residual**2/(2*variance)
    has_prior='mean' in target and 'covariance' in target
    if has_prior:
        diff=points-prior_mean;inv=np.linalg.pinv(prior_cov)
        logw-=0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
        old_lower,old_upper=radius_interval(points,positives[:-1],negatives)
        old_width=np.maximum(old_upper-old_lower,0.)
        likelihood=np.divide(new_width,old_width,out=np.zeros_like(new_width),where=old_width>1e-10)
    else:
        # 首次有效示向同时结合此前全部无信号信息，避免逐次独立重抽接收半径。
        likelihood=new_width/500.
    likelihood=np.where(np.isfinite(likelihood),likelihood,0.)
    valid=(distance>5)&(likelihood>0)&(np.linalg.norm(points,axis=1)<=1800+1e-6)
    for station in positives:valid &= np.linalg.norm(points-np.asarray(station),axis=1)>5
    failure_support=np.ones(len(points),dtype=bool)
    try:
        failed_history=normalize_points(target.get('failed_clears',[]))
    except (TypeError,ValueError):
        failed_history=np.empty((0,2),dtype=float);history_note='invalid_failure_history_fallback'
    for failed in failed_history:failure_support &= np.linalg.norm(points-failed,axis=1)>20
    valid &= failure_support
    note='gaussian_moment_projection'+((':'+history_note) if history_note else '')
    weights=None
    if np.any(valid):
        logw[valid]+=np.log(likelihood[valid]);logw[~valid]=-np.inf
        finite=np.isfinite(logw)
        if np.any(finite):
            max_log=float(np.max(logw[finite]))
            weights=np.zeros(len(points),dtype=float)
            weights[finite]=np.exp(np.clip(logw[finite]-max_log,-745.,709.))
    if weights is None or not math.isfinite(float(weights.sum())) or float(weights.sum())<=1e-300:
        # 极窄/边界区域可能被有限求积漏采；回到几何区域的保守矩近似，绝不判空。
        weights=areas*failure_support
        if weights.sum()<=0:weights=areas
        note='geometric_moment_fallback'+((':'+history_note) if history_note else '')
    try:
        mu,cov=gaussian_moments(points,weights)
    except (TypeError,ValueError,np.linalg.LinAlgError):
        mu,cov=_fallback_moments(poly,prior_mean,prior_cov)
        note='geometric_moment_fallback'+((':'+history_note) if history_note else '')
    target['mean']=mu;target['covariance']=cov
    return _posterior_record(mu,cov,note,len(points),noise_std)

def _legacy_update_gaussian_negative(target, old_negatives):
    """Condition a discovered target on one additional no-signal result.

    A known target remains an existing source; the observation only constrains
    its fixed reception radius.  The geometric polygon is intentionally kept
    as an outer bound, while the Gaussian planning posterior is updated with
    the ratio of new and old radius-compatible interval lengths.
    """
    triangles,areas=mesh(target['poly'],BAYES_CONFIG['quad_depth'])
    points=triangles.mean(axis=1)
    positives=target.get('stations',[])
    negatives=target.get('negatives',[])
    lower,old_upper=radius_interval(points,positives,old_negatives)
    _,new_upper=radius_interval(points,positives,negatives)
    old_width=np.maximum(old_upper-lower,0.)
    new_width=np.maximum(new_upper-lower,0.)
    diff=points-target['mean'];inv=np.linalg.inv(target['covariance'])
    logw=np.log(areas)-0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
    likelihood=np.divide(new_width,old_width,out=np.zeros_like(new_width),where=old_width>1e-10)
    valid=(likelihood>0)&(np.linalg.norm(points,axis=1)<=1800+1e-6)
    for station in positives:
        valid &= np.linalg.norm(points-np.asarray(station))>5
    failure_support=np.ones(len(points),dtype=bool)
    for failed in target.get('failed_clears',[]):
        failure_support &= np.linalg.norm(points-np.asarray(failed))>20
    valid &= failure_support
    note='negative_receive_conditioning'
    if np.any(valid):
        logw[valid]+=np.log(likelihood[valid]);logw[~valid]=-np.inf
        weights=np.exp(logw-np.max(logw))
    else:
        # Finite quadrature can miss a very thin surviving strip.  Keep a
        # geometric moment fallback and never turn this into a contradiction.
        weights=areas*((new_width>0)&failure_support)
        if weights.sum()<=0:weights=areas*failure_support
        if weights.sum()<=0:weights=areas
        note='geometric_negative_fallback'
    mu,cov=gaussian_moments(points,weights)
    target['mean']=mu;target['covariance']=cov
    return {'mean':mu.tolist(),'covariance':cov.tolist(),'note':note,
            'quadrature_points':len(points),
            'gaussian_entropy_nats':float(math.log(2*math.pi*math.e)+0.5*np.linalg.slogdet(cov)[1])}

def update_gaussian_negative(target, old_negatives):
    """Condition a discovered target on an additional no-signal result.

    This is a planning-only posterior update.  Empty/degenerate meshes,
    singular covariances, and underflowing likelihoods fall back to finite
    geometric moments; they never revoke the hard geometric target region.
    """
    if not isinstance(target,dict):
        raise ValueError('目标后验必须是字典')
    poly=target.get('poly',[])
    try:
        triangles,areas=mesh(poly,BAYES_CONFIG['quad_depth'])
    except (TypeError,ValueError,IndexError):
        triangles=np.empty((0,3,2),dtype=float);areas=np.empty(0,dtype=float)
    fallback_mean,fallback_cov=_fallback_moments(poly)
    prior_mean=_finite_point(target.get('mean'),fallback_mean)
    prior_cov=_project_covariance(target.get('covariance'))
    if prior_cov is None:prior_cov=fallback_cov.copy()
    history_note=''
    try:
        positives=normalize_points(target.get('stations',[]))
        negatives=normalize_points(target.get('negatives',[]))
        old_negative_values=normalize_points(old_negatives)
    except (TypeError,ValueError):
        positives=np.empty((0,2),dtype=float);negatives=np.empty((0,2),dtype=float)
        old_negative_values=np.empty((0,2),dtype=float)
        history_note='invalid_history_fallback'
    if not len(triangles):
        target['mean']=prior_mean;target['covariance']=prior_cov
        return _posterior_record(prior_mean,prior_cov,
                                 'degenerate_mesh_fallback'+(
                                     ':'+history_note if history_note else ''),0)
    points=triangles.mean(axis=1)
    lower,old_upper=radius_interval(points,positives,old_negative_values)
    _,new_upper=radius_interval(points,positives,negatives)
    old_width=np.maximum(old_upper-lower,0.)
    new_width=np.maximum(new_upper-lower,0.)
    diff=points-prior_mean;inv=np.linalg.pinv(prior_cov)
    logw=np.log(np.maximum(areas,np.finfo(float).tiny))-0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
    likelihood=np.divide(new_width,old_width,out=np.zeros_like(new_width),where=old_width>1e-10)
    likelihood=np.where(np.isfinite(likelihood),likelihood,0.)
    valid=(likelihood>0)&(np.linalg.norm(points,axis=1)<=1800+1e-6)
    for station in positives:
        valid &= np.linalg.norm(points-np.asarray(station),axis=1)>5
    failure_support=np.ones(len(points),dtype=bool)
    try:
        failed_history=normalize_points(target.get('failed_clears',[]))
    except (TypeError,ValueError):
        failed_history=np.empty((0,2),dtype=float);history_note='invalid_failure_history_fallback'
    for failed in failed_history:
        failure_support &= np.linalg.norm(points-failed,axis=1)>20
    valid &= failure_support
    note='negative_receive_conditioning'+((':'+history_note) if history_note else '')
    weights=None
    if np.any(valid):
        logw[valid]+=np.log(likelihood[valid]);logw[~valid]=-np.inf
        finite=np.isfinite(logw)
        if np.any(finite):
            max_log=float(np.max(logw[finite]))
            weights=np.zeros(len(points),dtype=float)
            weights[finite]=np.exp(np.clip(logw[finite]-max_log,-745.,709.))
    if weights is None or not math.isfinite(float(weights.sum())) or float(weights.sum())<=1e-300:
        # Finite quadrature can miss a very thin surviving strip.  Keep a
        # geometric moment fallback and never turn this into a contradiction.
        weights=areas*((new_width>0)&failure_support)
        if weights.sum()<=0:weights=areas*failure_support
        if weights.sum()<=0:weights=areas
        note='geometric_negative_fallback'+((':'+history_note) if history_note else '')
    try:
        mu,cov=gaussian_moments(points,weights)
    except (TypeError,ValueError,np.linalg.LinAlgError):
        mu,cov=_fallback_moments(poly,prior_mean,prior_cov)
        note='geometric_negative_fallback'+((':'+history_note) if history_note else '')
    target['mean']=mu;target['covariance']=cov
    return _posterior_record(mu,cov,note,len(points))


def update_unknown_belief(belief,coverage_map,ch):
    # 已存在目标的空间先验为圆内均匀；同一个固定R的全部no_signal联合似然。
    if not isinstance(belief,dict):
        raise ValueError('频道后验必须是字典')
    try:
        points=normalize_points(coverage_map.centers)
    except (AttributeError,TypeError,ValueError):
        points=np.empty((0,2),dtype=float)
    try:
        negatives=normalize_points(belief.get('negatives',[]))
    except (TypeError,ValueError):
        negatives=np.empty((0,2),dtype=float)
    if len(points):
        _,upper=radius_interval(points,[],negatives)
        weights=np.maximum(upper-1000.,0.)/500.
        inside=np.linalg.norm(points,axis=1)<=1800
        weights*=inside
    else:
        weights=np.empty(0,dtype=float);inside=np.empty(0,dtype=bool)
    evidence=float(weights.sum())/max(1,int(inside.sum()))
    if not math.isfinite(evidence):evidence=0.
    try:
        prior=float(BAYES_CONFIG['prior_exists'])
    except (TypeError,ValueError):
        prior=.65
    prior=min(max(prior,1e-9),1.-1e-9)
    denominator=1.-prior+prior*evidence
    probability=prior*evidence/denominator if denominator>1e-300 else 0.
    belief['existence']=min(1.,max(1e-9,float(probability)))
    note='negative_observation_moment_projection'
    if weights.sum()<=1e-12 and len(points):
        try:
            remaining=np.asarray(coverage_map.remaining,dtype=float)
            if (remaining.ndim==2 and 0<=int(ch)-1<remaining.shape[0]
                    and remaining.shape[1]==len(points)):
                weights=np.maximum(remaining[int(ch)-1],0.)
            else:
                weights=np.ones(len(points),dtype=float)
        except (AttributeError,TypeError,ValueError):
            weights=np.ones(len(points),dtype=float)
        note='grid_boundary_moment_fallback'
    if len(points) and float(np.sum(weights))>1e-12:
        try:
            belief['mean'],belief['covariance']=gaussian_moments(points,weights)
        except (TypeError,ValueError,np.linalg.LinAlgError):
            belief['mean'],belief['covariance']=_fallback_moments(
                None,belief.get('mean'),belief.get('covariance'))
            note='geometric_moment_fallback'
    else:
        belief['mean'],belief['covariance']=_fallback_moments(
            None,belief.get('mean'),belief.get('covariance'))
        note='empty_grid_moment_fallback'
    return {'mean':np.asarray(belief['mean'],dtype=float).tolist(),
            'covariance':np.asarray(belief['covariance'],dtype=float).tolist(),
            'existence_probability':belief['existence'],'note':note}

def binary_entropy(p):
    if p<=0 or p>=1:return 0.
    return -p*math.log(p)-(1-p)*math.log1p(-p)

def detection_information(belief,coverage_map,q):
    # P(收到信号)在均匀空间/半径先验及所有负观测条件下的数值近似。
    # H(检测结果)是关于完整潜在状态(存在性、位置、R)的二值观测信息代理。
    if not isinstance(belief,dict):return 0.,0.
    try:
        points=normalize_points(coverage_map.centers)
        q=np.asarray(q,dtype=float)
        if q.shape!=(2,) or not np.all(np.isfinite(q)):return 0.,0.
        negatives=normalize_points(belief.get('negatives',[]))
    except (AttributeError,TypeError,ValueError):
        return 0.,0.
    if not len(points):return 0.,0.
    _,upper=radius_interval(points,[],negatives)
    old=np.maximum(upper-1000.,0.)
    lower=np.maximum(1000.,np.linalg.norm(points-q,axis=1))
    new=np.maximum(upper-lower,0.)
    inside=np.linalg.norm(points,axis=1)<=1800
    old_total=float(old[inside].sum());new_total=float(new[inside].sum())
    a=float(new_total/max(1e-12,old_total)) if math.isfinite(new_total) else 0.
    try:existence=float(belief.get('existence',BAYES_CONFIG['prior_exists']))
    except (TypeError,ValueError):existence=0.
    probability=min(1.,max(0.,existence*a))
    return binary_entropy(probability),probability

def bayes_select(poly,stations,current,target,deadline):
    """局部高斯线性化下的预期熵下降/耗时；接收条件另作全域认证。"""
    begin=time.monotonic()
    failure_detail={'reason':'invalid_planning_input',
                    'reception_certificate':{'certified':False,
                        'scope':'entire_convex_polygon',
                        'reason':'invalid_or_degenerate_planning_input'}}
    if not isinstance(target,dict):return None,failure_detail
    try:
        poly=normalize_points(poly)
        stations=normalize_points(stations)
        current_values=normalize_points(current)
        if len(current_values)!=1:raise ValueError
        current=current_values[0]
        mu=np.asarray(target.get('mean'),dtype=float)
        cov=_project_covariance(target.get('covariance'))
        if mu.shape!=(2,) or not np.all(np.isfinite(mu)) or cov is None:raise ValueError
        noise_std=float(BAYES_CONFIG['noise_deg'])
        if not math.isfinite(noise_std) or noise_std<=0:raise ValueError
        variance=math.radians(noise_std)**2
        deadline=float(deadline)
        if not math.isfinite(deadline):deadline=math.inf
    except (TypeError,ValueError,OverflowError):
        return None,failure_detail
    scale=1000.
    try:
        triangles,_=mesh(poly,2)
    except (TypeError,ValueError,IndexError):
        triangles=np.empty((0,3,2),dtype=float)
    if not len(triangles):
        return None,{'reason':'degenerate_or_empty_polygon',
                     'reception_certificate':{'certified':False,
                         'scope':'entire_convex_polygon',
                         'reason':'polygon_has_no_positive_area_triangle'}}
    def receive(z):
        return reception_upper_bound_m2(triangles,z*scale,stations)/1e6
    def separation(z):return float(np.linalg.norm(z*scale-mu)-20)/scale
    def station_gap(z):
        if not len(stations):return math.inf
        return float(np.min(np.linalg.norm(z*scale-stations,axis=1))-5)/scale
    def information(z):
        delta=mu-z*scale;r2=float(delta@delta)
        if r2<1e-8:return 0.
        h=np.array([-delta[1],delta[0]])/r2
        return 0.5*math.log1p(max(0.,float(h@cov@h))/variance)
    def duration(z):
        p=z*scale
        return 10+np.linalg.norm(p-current)/5+0.5*np.linalg.norm(p-mu)/5
    def objective(z):
        if time.monotonic()>deadline-20:raise BudgetStop('信息增益规划接近现实截止时间。')
        return -information(z)/duration(z)
    def feasible(z):
        # Keep seed filtering and the optimizer constraint on the same strict
        # squared-distance certificate.  The previous scaled tolerance was
        # looser than the final certificate and could admit a boundary seed.
        return (reception_certified(receive(z)*scale**2) and
                separation(z)>=-1e-8 and station_gap(z)>=-1e-8)
    center,radius=mec(poly);center=np.asarray(center)
    seeds=[]
    for origin in (mu,center):
        for radius0 in sorted(set([25.,50.,100.,200.,min(500.,max(50.,radius*.5))])):
            for angle in np.arange(12)*math.pi/6:
                z=(origin+radius0*np.array([math.cos(angle),math.sin(angle)]))/scale
                if feasible(z):seeds.append((objective(z),z))
    if not seeds:return None,{'reason':'no_feasible_information_point',
        'reception_certificate':{'certified':False,'scope':'entire_convex_polygon',
        'reason':'no_candidate_satisfied_hard_reception_constraint'}}
    seeds.sort(key=lambda x:x[0]);pool=list(seeds);starts=[]
    for value,z in seeds:
        if not starts or all(np.linalg.norm(z-p)>0.1 for p in starts):starts.append(z)
        if len(starts)==4:break
    anchor=triangles[0].mean(axis=0)
    station_bound=(float(np.max(np.linalg.norm(stations-anchor,axis=1)))
                   if len(stations) else 0.)
    bound=max(RECEIVE_RADIUS_M,station_bound)
    box=[((anchor[k]-bound)/scale,(anchor[k]+bound)/scale) for k in range(2)]
    reception_margin_scaled=RECEIVE_MARGIN_M2/scale**2
    constraints=[{'type':'ineq','fun':lambda z:-receive(z)-reception_margin_scaled},
                 {'type':'ineq','fun':separation},{'type':'ineq','fun':station_gap}]
    for z in starts:
        try:
            result=minimize(objective,z,method='SLSQP',bounds=box,constraints=constraints,
                            options={'maxiter':60,'ftol':1e-8})
        except BudgetStop:
            raise
        except Exception:
            # Optimizer setup/evaluation can fail for a nearly singular
            # posterior or an ill-conditioned boundary.  The deterministic
            # feasible seeds are still valid, so simply retain them as the
            # safe fallback and continue with the remaining starts.
            continue
        if (getattr(result,'success',False) and np.all(np.isfinite(result.x))
                and feasible(result.x)):
            pool.append((objective(result.x),result.x))
    original_value,original_z=min(pool,key=lambda x:x[0])
    original_score=max(0.,-float(original_value));original_point=tuple(float(v) for v in original_z*scale)
    # DOP is only defined once a real bearing station exists.  A caller may
    # legitimately pass an empty station list during the first planning
    # round; keep the primary information objective in that case and skip
    # the secondary tie-break rather than indexing an empty list.
    target_stations=normalize_points(target.get('stations',[]))
    original_dop=(depth_weighted_dop(poly,target_stations[0],original_point)
                  if len(target_stations) else math.inf)
    chosen_value,chosen_z=original_value,original_z
    chosen_dop=original_dop;override=False
    cfg=Q12_LINK_CONFIG;tie_candidates=[]
    for candidate_value,candidate_z in pool:
        score=max(0.,-float(candidate_value))
        primary_loss=(original_score-score)/max(original_score,1e-12)
        if primary_loss<=min(cfg['tau'],cfg['max_primary_loss_ratio'],cfg['primary_tie_rel_tol'])+1e-15:
            p=tuple(float(v) for v in candidate_z*scale)
            if not len(target_stations):
                continue
            dop=depth_weighted_dop(poly,target_stations[0],p)
            tie_candidates.append((dop,primary_loss,candidate_value,candidate_z))
    finite=[x for x in tie_candidates if math.isfinite(x[0])]
    if finite and math.isfinite(original_dop):
        best_dop=min(x[0] for x in finite)
        eligible=[x for x in finite if x[0]<=best_dop*(1+cfg['tau'])]
        dop,primary_loss,candidate_value,candidate_z=min(eligible,key=lambda x:(x[0],x[1]))
        improvement=(original_dop-dop)/max(original_dop,1e-12)
        if (cfg['enable_dop_override'] and improvement>=cfg['min_dop_improvement'] and
                primary_loss<=cfg['max_primary_loss_ratio']):
            chosen_dop,chosen_value,chosen_z=dop,candidate_value,candidate_z;override=True
    point=tuple(float(v) for v in chosen_z*scale)
    primary_loss=(original_score-max(0.,-float(chosen_value)))/max(original_score,1e-12)
    improvement=((original_dop-chosen_dop)/max(original_dop,1e-12)
                 if math.isfinite(original_dop) and math.isfinite(chosen_dop) else 0.)
    receive_upper=receive(chosen_z)*1e6
    reception_certificate={'certified':reception_certified(receive_upper),
        'scope':'entire_convex_polygon','method':'convex_quadratic_maximum_on_triangle_vertices',
        'receive_upper_m2':receive_upper,'constraint':'candidate_receives_for_every_G_in_polygon'}
    tiebreak={'original_position':original_point,'chosen_position':point,
        'original_primary_objective':float(original_value),'chosen_primary_objective':float(chosen_value),
        'original_dop':original_dop if math.isfinite(original_dop) else None,
        'chosen_dop':chosen_dop if math.isfinite(chosen_dop) else None,
        'relative_dop_improvement':improvement,'primary_loss_ratio':primary_loss,
        'candidate_count':len(pool),'near_primary_candidate_count':len(tie_candidates),
        'override':override,'reception_certificate':reception_certificate}
    return point,{'reason':'selected','position':point,'expected_information_nats':information(chosen_z),
                  'information_per_second':-float(chosen_value),'estimated_action_chain_s':float(duration(chosen_z)),
                  'receive_upper_m2':receive_upper,'elapsed_s':time.monotonic()-begin,
                  'reception_certificate':reception_certificate,'problem2_dop_tiebreak':tiebreak,
                  'note':'linearized_gaussian_information_surrogate_with_hard_reception_certificate'}

ROUTE_MEASURE_CONFIG={'min_information_ratio':0.70,'min_saving_s':10.0,
                      'segment_fractions':(0.25,0.5,0.75,1.0),
                      'min_stop_information_nats':0.50,'max_target_channels_per_stop':6}

def mandatory_stop_measurement_score(target,q):
    """评价必经扫描点是否适合作为已发现目标的顺路补测点。"""
    if not isinstance(target,dict) or 'mean' not in target or 'covariance' not in target:
        return None
    try:
        q=np.asarray(q,dtype=float)
        if q.shape!=(2,) or not np.all(np.isfinite(q)):
            return None
        poly=normalize_points(target.get('poly'))
        stations=normalize_points(target.get('stations',[]))
    except (TypeError,ValueError):
        return None
    if len(poly)<3:
        return None
    triangles,_=mesh(poly,2)
    if not len(triangles):
        return None
    receive_upper=reception_upper_bound_m2(triangles,q,stations)
    if not reception_certified(receive_upper):return None
    if len(stations) and float(np.min(np.linalg.norm(stations-q,axis=1)))<5-1e-7:return None
    try:
        mu=np.asarray(target['mean'],dtype=float)
        covariance=_project_covariance(target['covariance'])
        if mu.shape!=(2,) or not np.all(np.isfinite(mu)) or covariance is None:
            return None
        noise_std=float(BAYES_CONFIG['noise_deg'])
        if not math.isfinite(noise_std) or noise_std<=0:return None
    except (TypeError,ValueError):
        return None
    delta=mu-q;r2=float(delta@delta)
    if r2<20**2:return None
    h=np.array([-delta[1],delta[0]])/r2
    variance=math.radians(noise_std)**2
    info=0.5*math.log1p(max(0.,float(h@covariance@h))/variance)
    if not math.isfinite(info):return None
    if info<ROUTE_MEASURE_CONFIG['min_stop_information_nats']:return None
    receive_bound=float(math.sqrt(max(0.,np.max(np.sum((triangles-q)**2,axis=2)))))
    return {'information_nats':info,'receive_bound_m':receive_bound,
            'receive_upper_m2':float(receive_upper),
            'reception_certificate':{'certified':True,
                'scope':'entire_convex_polygon',
                'method':'convex_quadratic_maximum_on_triangle_vertices',
                'receive_upper_m2':float(receive_upper)}}

def route_aligned_measurement(poly,stations,current,target,next_stop,free_point,free_detail):
    """在当前点到下一必经点的线段上寻找可替代自由补测点。

    仅当候选点满足保守接收条件、与既有测站/目标估计保持间距、信息量达到
    自由最优点的指定比例，并且预计“补测后去中心再回到下一节点”的总链路
    至少节省指定秒数时采用。这样不会仅为贴路线而牺牲过多交会质量。
    """
    if next_stop is None or free_point is None:return free_point,{**free_detail,'route_aligned':False,'route_reason':'no_next_stop'}
    if not isinstance(target,dict) or 'mean' not in target or 'covariance' not in target:
        return tuple(free_point),{**free_detail,'route_aligned':False,
                                  'route_reason':'invalid_target_posterior'}
    try:
        vertices=normalize_points(poly)
        stations=normalize_points(stations)
        current_values=normalize_points(current)
        next_stop_values=normalize_points(next_stop)
        free_values=normalize_points(free_point)
    except (TypeError,ValueError):
        return tuple(free_point),{**free_detail,'route_aligned':False,
                                  'route_reason':'invalid_geometry_or_point_input'}
    if (len(vertices)<3 or len(current_values)!=1 or len(next_stop_values)!=1 or
            len(free_values)!=1):
        return tuple(free_point),{**free_detail,'route_aligned':False,
                                  'route_reason':'degenerate_or_empty_polygon'}
    current=current_values[0];next_stop=next_stop_values[0];free_point=free_values[0]
    try:
        mu=np.asarray(target['mean'],dtype=float);cov=np.asarray(target['covariance'],dtype=float)
        if mu.shape!=(2,) or cov.shape!=(2,2) or not np.all(np.isfinite(mu)) or not np.all(np.isfinite(cov)):
            raise ValueError
    except (TypeError,ValueError):
        return tuple(free_point),{**free_detail,'route_aligned':False,
                                  'route_reason':'invalid_target_posterior'}
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    triangles,_=mesh(vertices,2)
    if not len(triangles):
        return tuple(free_point),{**free_detail,'route_aligned':False,
                                  'route_reason':'degenerate_or_empty_polygon',
                                  'reception_certificate':{'certified':False,
                                      'scope':'entire_convex_polygon',
                                      'reason':'polygon_has_no_positive_area_triangle'}}
    def receive(p):
        return reception_upper_bound_m2(triangles,p,stations)
    def information(p):
        delta=mu-p;r2=float(delta@delta)
        if r2<1e-8:return 0.
        h=np.array([-delta[1],delta[0]])/r2
        return 0.5*math.log1p(max(0.,float(h@cov@h))/variance)
    def feasible(p):
        station_ok=(not len(stations) or
                    float(np.min(np.linalg.norm(p-stations,axis=1)))>=5-1e-7)
        return (reception_certified(receive(p)) and
                np.linalg.norm(p-mu)>=20-1e-7 and station_ok)
    def chain_seconds(p):
        # 补测5秒；频道切换对自由点和顺路点相同，比较时可抵消。
        return (np.linalg.norm(p-current)+np.linalg.norm(p-mu)+np.linalg.norm(mu-next_stop))/5+5
    free_info=float(free_detail.get('expected_information_nats',information(free_point)))
    free_chain=chain_seconds(free_point)
    candidates=[]
    delta=next_stop-current
    for fraction in ROUTE_MEASURE_CONFIG['segment_fractions']:
        p=current+float(fraction)*delta
        if feasible(p):
            info=information(p);chain=chain_seconds(p)
            candidates.append((chain, -info, float(fraction), p, info))
    if not candidates:
        return tuple(free_point),{**free_detail,'route_aligned':False,'route_reason':'no_feasible_route_candidate'}
    chain,_,fraction,p,info=min(candidates,key=lambda x:(x[0],x[1]))
    saving=free_chain-chain
    ratio=info/max(free_info,1e-12)
    if ratio>=ROUTE_MEASURE_CONFIG['min_information_ratio'] and saving>=ROUTE_MEASURE_CONFIG['min_saving_s']:
        point=tuple(float(v) for v in p)
        return point,{'reason':'route_aligned_selected','position':point,
                      'expected_information_nats':info,'free_expected_information_nats':free_info,
                      'information_ratio':ratio,'estimated_chain_s':chain,
                      'free_estimated_chain_s':free_chain,'estimated_saving_s':saving,
                      'segment_fraction':fraction,'next_stop':tuple(float(v) for v in next_stop),
                      'route_aligned':True,
                      'note':'feasible_point_on_current_to_next_planned_stop'}
    return tuple(free_point),{**free_detail,'route_aligned':False,
                              'route_reason':'insufficient_information_or_saving',
                              'best_route_information_ratio':ratio,
                              'best_route_estimated_saving_s':saving,
                              'next_stop':tuple(float(v) for v in next_stop)}

PROBE_CONFIG={'max_attempts':5,'min_probability':0.2,'max_radius':80.}

JOINT_CONFIG={'max_detour_m':5.,'horizon':3,'min_saving_s':0.,'candidate_limit':12}

def _legacy_probe_distribution(target):
    """失败clear后保留条件分布的非高斯形状，不反复用高斯投影重加权。
    新测向到来时(测点数改变)重新以新高斯工作后验生成求积权重。
    """
    version=len(target.get('stations',[]))
    cached=target.get('probe_base')
    if cached is None or cached['version']!=version:
        triangles,areas=mesh(target['poly'],4);points=triangles.mean(axis=1)
        if not len(points):
            return None
        diff=points-target['mean'];inv=np.linalg.inv(target['covariance'])
        logw=np.log(areas)-0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
        weights=np.exp(logw-np.max(logw));weights/=weights.sum()
        cached={'version':version,'points':points,'weights':weights};target['probe_base']=cached
    points=cached['points'];weights=cached['weights'].copy()
    for q in target.get('failed_clears',[]):weights[np.linalg.norm(points-q,axis=1)<=20]=0.
    if weights.sum()<=1e-300:return None
    weights/=weights.sum()
    return points,weights

def probe_distribution(target):
    """Return a finite quadrature posterior for probe planning, or ``None``.

    Invalid/degenerate geometry and singular covariance are ordinary planner
    edge cases; callers then fall back to the certified grid path.
    """
    if not isinstance(target,dict):
        return None
    try:
        version=len(normalize_points(target.get('stations',[])))
        poly=normalize_points(target.get('poly',[]))
        if len(poly)<3:return None
        mean=_finite_point(target.get('mean'))
        covariance=_project_covariance(target.get('covariance'))
        if covariance is None:return None
        triangles,areas=mesh(poly,4)
    except (TypeError,ValueError,IndexError):
        return None
    if not len(triangles) or not np.all(np.isfinite(areas)) or np.any(areas<=0):
        return None
    cached=target.get('probe_base')
    if (not isinstance(cached,dict) or cached.get('version')!=version or
            not isinstance(cached.get('points'),np.ndarray) or
            not isinstance(cached.get('weights'),np.ndarray)):
        points=triangles.mean(axis=1)
        diff=points-mean;inv=np.linalg.pinv(covariance)
        logw=np.log(np.maximum(areas,np.finfo(float).tiny))-0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
        finite=np.isfinite(logw)
        if not np.any(finite):return None
        max_log=float(np.max(logw[finite]));weights=np.zeros(len(points),dtype=float)
        weights[finite]=np.exp(np.clip(logw[finite]-max_log,-745.,709.))
        total=float(weights.sum())
        if not math.isfinite(total) or total<=1e-300:return None
        weights/=total
        cached={'version':version,'points':points,'weights':weights};target['probe_base']=cached
    points=np.asarray(cached['points'],dtype=float);weights=np.asarray(cached['weights'],dtype=float).copy()
    if points.ndim!=2 or points.shape[1]!=2 or len(points)!=len(weights):return None
    try:
        failed_history=normalize_points(target.get('failed_clears',[]))
    except (TypeError,ValueError):
        return None
    for q in failed_history:weights[np.linalg.norm(points-q,axis=1)<=20]=0.
    total=float(weights.sum())
    if not math.isfinite(total) or total<=1e-300:return None
    weights/=total
    return points,weights


def joint_probe_plan(target,current,current_channel,ch,anchor,same_target,next_stop):
    """在实际即将执行的路段上插入1~3次clear，枚举成功/全部失败分支。
    返回值为启发式预计节省，不是无条件时间上界；只执行第一步，反馈后滚动重算。
    """
    if not isinstance(target,dict):return None
    try:
        remaining_budget=int(PROBE_CONFIG['max_attempts'])-int(target.get('probe_attempts',0))
        radius=float(target.get('radius',math.inf))
        if not math.isfinite(radius) or radius<0.:return None
        current_values=normalize_points(current)
        anchor_values=normalize_points(anchor)
        if len(current_values)!=1 or len(anchor_values)!=1:return None
        current=current_values[0];anchor=anchor_values[0]
        if next_stop is not None:
            next_stop_values=normalize_points(next_stop)
            if len(next_stop_values)!=1:return None
            next_stop=next_stop_values[0]
        # Validate all target coordinates before constructing projected seeds.
        # ``_finite_point`` deliberately supplies a fallback for posterior
        # updates, but a probe plan must not silently turn malformed caller
        # input into a real action.
        mean_value=np.asarray(target.get('mean'),dtype=float)
        center_value=np.asarray(target.get('center'),dtype=float)
        planned_value=np.asarray(target.get('planned'),dtype=float)
        if (mean_value.shape!=(2,) or center_value.shape!=(2,) or
                planned_value.shape!=(2,) or
                not np.all(np.isfinite(mean_value)) or
                not np.all(np.isfinite(center_value)) or
                not np.all(np.isfinite(planned_value))):
            return None
        failed_history=normalize_points(target.get('failed_clears',[]))
        lookahead=float(SCHEDULE_CONFIG['lookahead'])
        if (not math.isfinite(lookahead) or lookahead<0. or lookahead>1.):
            return None
    except (TypeError,ValueError,OverflowError):
        return None
    if remaining_budget<=0 or radius>PROBE_CONFIG['max_radius'] or target.get('planned') is None:return None
    sample=probe_distribution(target)
    if sample is None:return None
    points,weights=sample
    if (points.ndim!=2 or points.shape[1]!=2 or weights.ndim!=1 or
            len(points)!=len(weights) or not len(points)):
        return None
    vector=anchor-current;length2=float(vector@vector);direct=dist(current,anchor)
    def project(q):
        if length2<=1e-12:return current.copy()
        return current+np.clip(float((np.asarray(q)-current)@vector)/length2,0,1)*vector
    seeds=[tuple(current),tuple(project(mean_value)),tuple(project(center_value))]
    if length2>1e-12:
        direction=vector/math.sqrt(length2);center=project(mean_value)
        for offset in (-40,-20,-10,10,20,40):seeds.append(tuple(project(center+offset*direction)))
    try:
        poly=normalize_points(target.get('poly',[]))
        if len(poly)<3:return None
        low=poly.min(axis=0);high=poly.max(axis=0)
    except (TypeError,ValueError):
        return None
    n=np.maximum(1,np.ceil((high-low)/25).astype(int))
    seeds.extend([tuple(mean_value),tuple(center_value)])
    for iy in range(n[1]):
        for ix in range(n[0]):
            q=low+(np.array([ix,iy])+.5)*(high-low)/n
            seeds.extend([tuple(q),tuple(project(q))])
    candidates=[];seen=set()
    for q in seeds:
        key=tuple(round(x,5) for x in q)
        if key in seen:continue
        seen.add(key)
        if any(dist(q,p)<1 for p in failed_history):continue
        extra=dist(current,q)+dist(q,anchor)-direct
        if extra>JOINT_CONFIG['max_detour_m']+1e-7:continue
        mask=np.linalg.norm(points-q,axis=1)<=20
        mass=float(weights[mask].sum())
        if mass<0.01:continue
        candidates.append({'q':tuple(q),'mask':mask,'mass':mass,'extra':max(0.,extra)})
    candidates.sort(key=lambda c:c['mass']/(3+dist(current,c['q'])/5),reverse=True)
    candidates=candidates[:JOINT_CONFIG['candidate_limit']]
    if not candidates:return None
    planned=planned_value
    center=center_value
    if same_target:
        # baseline: current->测向点->估计清除点->next_stop。
        tail=5+(current_channel!=ch)+dist(anchor,center)/5+5
        downstream=dist(center,next_stop)/5 if next_stop is not None else 0.
        success_tail=lambda q:lookahead*dist(q,next_stop)/5 if next_stop is not None else 0.
    else:
        # 顺路处理其他目标：保留既定路线，估算避免将来专门定位该目标的成本。
        tail=dist(anchor,planned)/5+5+(current_channel!=ch)+dist(planned,center)/5+5
        downstream=0.
        success_tail=lambda q:lookahead*dist(q,anchor)/5
    baseline=direct/5+tail+lookahead*downstream
    best=None;max_depth=min(JOINT_CONFIG['horizon'],remaining_budget)
    def visit(order,excluded,prob_remaining,travel_so_far,action_expected,last,success_return):
        nonlocal best
        if len(order)>=max_depth:return
        for i,candidate in enumerate(candidates):
            if i in order:continue
            q=candidate['q'];distance=dist(last,q)
            route_length=travel_so_far+distance
            detour=route_length+dist(q,anchor)-direct
            # 对“全部失败”的完整插入路径限制额外路程，而非分别限制每一个点。
            if detour>JOINT_CONFIG['max_detour_m']+1e-7:continue
            fresh=candidate['mask']&~excluded
            probability=.995*float(weights[fresh].sum()) # 预留求积遗漏尾部
            if probability<=1e-6:continue
            conditional=probability/prob_remaining
            if not order and conditional<PROBE_CONFIG['min_probability']:continue
            if order and conditional<0.05:continue
            after=max(.005,prob_remaining-probability)
            expected=action_expected+prob_remaining*(distance/5+3)+2*probability
            returned=success_return+probability*success_tail(q)
            total=expected+returned+after*(dist(q,anchor)/5+tail+lookahead*downstream)
            sequence=order+[i]
            saving=baseline-total
            if saving>=JOINT_CONFIG['min_saving_s'] and (best is None or saving>best['expected_saving_s']):
                best={'position':candidates[sequence[0]]['q'],
                      'sequence':[candidates[j]['q'] for j in sequence],
                      'probability':.995*candidates[sequence[0]]['mass'],
                      'sequence_success_probability':1-after,'expected_time_s':total,
                      'reference_time_s':baseline,'expected_saving_s':saving,
                      'worst_branch_extra_distance_m':max(0.,detour),'anchor':tuple(anchor),
                       'same_target':same_target,'lookahead_weight':lookahead,
                       'hypothesis':'approximate_posterior_and_one_measurement_continuation'}
            visit(sequence,excluded|candidate['mask'],after,route_length,expected,q,returned)
    visit([],np.zeros(len(points),dtype=bool),1.,0.,0.,tuple(current),0.)
    return best

def condition_failed_clear(target,q):
    q=tuple(float(v) for v in q)
    target.setdefault('failed_clears',[]).append(q)
    if target.get('fallback') is not None:
        # A previously built grid can contain points made unsafe by this new
        # observation. Remove them before the next fallback attempt.
        target['fallback']=[p for p in target['fallback'] if dist(p,q)>20+1e-7]
    # The cached quadrature was built from the pre-failure Gaussian.  It must
    # not survive the posterior conditioning, even though the station count
    # (the cache version key) is unchanged.
    target.pop('probe_base',None)
    target['planned']=None
    sample=probe_distribution(target)
    if sample is None:
        target.pop('probe_base',None)
        return {'note':'quadrature_missed_remaining_region_keep_geometric_outer_bound'}
    target['mean'],target['covariance']=gaussian_moments(*sample)
    # The sample above conditions the old posterior; rebuild it lazily from
    # the new moments on the next probe-planning call.
    target.pop('probe_base',None)
    return {'note':'posterior_conditioned_on_distance_greater_than_20',
            'mean':target['mean'].tolist(),'covariance':target['covariance'].tolist()}

LAYOUT_PRESETS = {
    'ring6_1123': {'ring_points': 6, 'ring_radius': 1123.0},
    'ring7_998': {'ring_points': 7, 'ring_radius': 998.0},
    'ring8_939': {'ring_points': 8, 'ring_radius': 939.0},
    'ring9_904': {'ring_points': 9, 'ring_radius': 904.0},
    'ring10_881': {'ring_points': 10, 'ring_radius': 881.0},
}
DEFAULT_LAYOUT = 'ring8_939'

def scan_points(ring_points, ring_radius):
    """返回原点及均匀分布在指定半径圆周上的扫描点。"""
    return [(0., 0.)] + [
        (ring_radius*math.cos(2*math.pi*k/ring_points),
         ring_radius*math.sin(2*math.pi*k/ring_points))
        for k in range(ring_points)
    ]

def coverage_certificate(ring_points=8, ring_radius=939.):
    """解析认证原点+均匀环点对1800米圆域的1000米覆盖。

    原点覆盖半径1000米的内盘。外环中任一点与最近环点夹角不超过
    pi/n；距离平方关于目标极径为凸函数，因此最坏值出现在1000或
    1800米端点。该结论适用于任意 n>=2。
    """
    if isinstance(ring_points, bool) or not isinstance(ring_points, int) or ring_points < 2:
        raise ValueError('ring_points 必须是大于等于2的整数')
    if not math.isfinite(ring_radius) or ring_radius <= 0:
        raise ValueError('ring_radius 必须是有限正数')
    half_sector = math.pi/ring_points
    endpoint_bounds = {
        str(r): math.sqrt(r*r+ring_radius*ring_radius-
                          2*r*ring_radius*math.cos(half_sector))
        for r in (1000., 1800.)
    }
    outer = max(endpoint_bounds.values())
    whole = max(1000., outer)
    if outer > 1000.+1e-9:
        raise ValueError(
            f'布局不能保证覆盖：n={ring_points}, R={ring_radius:g}米，'
            f'外环最不利距离={outer:.6f}米 > 1000米')
    return {'target_radius_m':1800.,'guaranteed_reception_radius_m':1000.,
            'ring_points':ring_points,'ring_radius_m':float(ring_radius),
            'half_sector_angle_deg':math.degrees(half_sector),
            'endpoint_distance_bounds_m':endpoint_bounds,
            'outer_annulus_distance_bound_m':outer,'whole_domain_bound_m':whole,
            'certified':True,
            'method':'analytic_origin_disk_and_uniform_ring_angular_sectors'}

def resolve_layout(layout_name=DEFAULT_LAYOUT, ring_points=None, ring_radius=None):
    preset = LAYOUT_PRESETS[layout_name]
    n = preset['ring_points'] if ring_points is None else ring_points
    radius = preset['ring_radius'] if ring_radius is None else ring_radius
    custom = ring_points is not None or ring_radius is not None
    name = layout_name if not custom else f'custom_n{n}_R{radius:g}'
    certificate = coverage_certificate(n, radius)
    return {'layout_name':name,'base_preset':layout_name,'ring_points':n,
            'ring_radius':float(radius),'certificate':certificate}

def rolling_open_route(current,nodes):
    """访问全部当前任务点的开放欧氏路径；服务时间在本轮固定，不影响排序。
    <=10个非当前位置节点时Held-Karp精确；更大时多初值最近邻+2-opt近似。
    只决定本轮下一动作，每次响应后重新建节点，不承诺未知环境全局最优。
    """
    if not nodes:return [],{'method':'empty','length_m':0.,'exact':True}
    zero=[i for i,n in enumerate(nodes) if dist(current,n['position'])<1e-6]
    ids=[i for i in range(len(nodes)) if i not in zero]
    n=len(ids)
    if not n:return zero,{'method':'at_current_position','length_m':0.,'exact':True}
    xy=np.asarray([current]+[nodes[i]['position'] for i in ids],dtype=float)
    d=np.linalg.norm(xy[:,None,:]-xy[None,:,:],axis=2)
    if n<=10:
        count=1<<n;dp=np.full((count,n),np.inf);parent=np.full((count,n),-1,dtype=int)
        for j in range(n):dp[1<<j,j]=d[0,j+1]
        for mask in range(1,count):
            for last in range(n):
                if not(mask>>last&1):continue
                prev=mask^(1<<last)
                if not prev:continue
                candidates=[j for j in range(n) if prev>>j&1]
                values=dp[prev,candidates]+d[np.asarray(candidates)+1,last+1]
                k=int(np.argmin(values));dp[mask,last]=values[k];parent[mask,last]=candidates[k]
        mask=count-1;last=int(np.argmin(dp[mask]));length=float(dp[mask,last]);order=[]
        while mask:
            order.append(last);old=int(parent[mask,last]);mask^=1<<last;last=old
        order.reverse();method='held_karp_exact';exact=True
    else:
        best=None
        for first in np.argsort(d[0,1:])[:min(n,6)]:
            order=[int(first)];remaining=set(range(n))-{int(first)}
            while remaining:
                nxt=min(remaining,key=lambda j:d[order[-1]+1,j+1]);order.append(nxt);remaining.remove(nxt)
            for _ in range(40):
                gain=0.;pair=None
                for i in range(n-1):
                    a=0 if i==0 else order[i-1]+1;b=order[i]+1
                    for j in range(i+1,n):
                        c=order[j]+1
                        delta=d[a,b]-d[a,c]
                        if j+1<n:
                            end=order[j+1]+1;delta+=d[c,end]-d[b,end]
                        if delta>gain+1e-9:gain=delta;pair=(i,j)
                if pair is None:break
                i,j=pair;order[i:j+1]=reversed(order[i:j+1])
            length=float(d[0,order[0]+1]+sum(d[a+1,b+1] for a,b in zip(order,order[1:])))
            if best is None or length<best[0]:best=(length,list(order))
        length,order=best;method='multi_start_2opt_approximate';exact=False
    return zero+[ids[i] for i in order],{'method':method,'length_m':length,'exact':exact,'task_nodes':len(nodes)}

class CoverageMap:
    """每频道独立的保守剩余区域地图。
    与目标圆相交的方格全部保留；no_signal只删除四角均在1000米内的格。
    方格全部删除=>该频道无目标。格心仅用于选址，不作为真实目标离散位置。
    """
    def __init__(self, cell_size=100.):
        self.n=math.ceil(3600/cell_size)
        self.step=3600/self.n
        axis=-1800+np.arange(self.n)*self.step
        x,y=np.meshgrid(axis,axis)
        lo=np.column_stack([x.ravel(),y.ravel()]); hi=lo+self.step
        nearest=np.maximum(np.maximum(lo,-hi),0.)
        self.valid=np.sum(nearest**2,axis=1)<=1800**2+1e-6
        self.ids=np.flatnonzero(self.valid)
        self.centers=(lo[self.valid]+hi[self.valid])/2
        self.corners=np.stack([lo[self.valid],lo[self.valid]+[self.step,0],
                               hi[self.valid],lo[self.valid]+[0,self.step]],axis=1)
        self.remaining=np.ones((20,len(self.ids)),dtype=bool)
        self.observations=np.zeros(20,dtype=int)
    def covered(self,q):
        return np.max(np.sum((self.corners-np.asarray(q))**2,axis=2),axis=1)<=(1000-1e-5)**2
    def update_negative(self,ch,q):
        before=int(self.remaining[ch-1].sum())
        self.remaining[ch-1,self.covered(q)]=False
        self.observations[ch-1]+=1
        return before-int(self.remaining[ch-1].sum())
def plan_cpp(coverage_map):
    """有限候选布局上的覆盖选点+开放路径动态规划。
    圆域由保守方格外包，只有整格被某扫描点1000米圆盘覆盖才认证。
    不是在移动途中连续感知；也不声称连续平面上的全局最优。
    """
    families=[('raster_3x3',[(0.,0.)]+[(float(x),float(y)) for y in (-1200,0,1200)
                                     for x in (-1200,0,1200) if (x,y)!=(0,0)])]
    for radius in (1250,1300,1350,1400,1450,1500):
        for offset in (0.,math.pi/6):
            families.append((f'hex_{radius}_{round(math.degrees(offset))}',[(0.,0.)]+
                [(radius*math.cos(offset+k*math.pi/3),radius*math.sin(offset+k*math.pi/3)) for k in range(6)]))
    records=[];best=None
    for name,points in families:
        covers=[coverage_map.covered(p) for p in points]
        if not np.logical_or.reduce(covers).all():
            records.append({'layout':name,'certified':False});continue
        n=len(points)-1; count=1<<n
        # dp[mask,j]:从原点出发访问mask，最后停在点j+1的最短开放路径。
        dp=np.full((count,n),np.inf);parent=np.full((count,n),-1,dtype=int)
        union=[None]*count;union[0]=covers[0].copy()
        for j in range(n):dp[1<<j,j]=dist(points[0],points[j+1])
        best_family=None
        for mask in range(1,count):
            bit=mask&-mask;j=bit.bit_length()-1
            union[mask]=union[mask^bit]|covers[j+1]
            for last in range(n):
                if not(mask>>last&1) or not math.isfinite(dp[mask,last]):continue
                for nxt in range(n):
                    if mask>>nxt&1:continue
                    newmask=mask|(1<<nxt)
                    value=dp[mask,last]+dist(points[last+1],points[nxt+1])
                    if value<dp[newmask,nxt]:
                        dp[newmask,nxt]=value;parent[newmask,nxt]=last
            if not union[mask].all():continue
            last=int(np.argmin(dp[mask]));length=float(dp[mask,last])
            # 每个检测点都可从当前频道开始：20次测量耗时20*5秒，
            # 仅其余19个频道需要切换，故每点均为119秒。
            # 这是统一比较布局的代理耗时；实际会按频道状态减少扫描。
            scan_time=119*(1+bin(mask).count('1'))
            score=length/5+scan_time
            if best_family is None or score<best_family[0]:best_family=(score,mask,last,length,scan_time)
        if best_family is None:raise RuntimeError('覆盖布局动态规划失败')
        score,mask,last,length,scan_time=best_family
        order=[];work=mask
        while work:
            order.append(last+1);previous=int(parent[work,last]);work^=1<<last;last=previous
        route=[points[0]]+[points[i] for i in reversed(order)]
        record={'layout':name,'certified':True,'waypoint_count':len(route),'route_length_m':length,
                'full_scan_surrogate_s':score,'scan_time_surrogate_s':scan_time}
        records.append(record)
        if best is None or score<best['full_scan_surrogate_s']:
            best={**record,'waypoints':route,'coverage_cells':len(coverage_map.ids)}
    if best is None:raise RuntimeError('没有通过全域覆盖认证的CPP布局')
    best['evaluated_layouts']=records
    return best

class HttpBackend:
    def __init__(self,url):
        self.url=url.rstrip('/')
        self.opener=build_opener(ProxyHandler({})) # 本机接口不经过系统代理
    def post(self,path,payload,timeout):
        data=json.dumps(payload,ensure_ascii=False,allow_nan=False).encode('utf-8')
        req=Request(self.url+path,data=data,headers={'Content-Type':'application/json'},method='POST')
        with self.opener.open(req,timeout=timeout) as response:
            if response.status!=200:
                raise RuntimeError('非200 HTTP响应')
            return json.loads(response.read().decode('utf-8'))

class LocalBackend:
    """独立生成10~16个全向源；同频道同位置误差固定，无真值传给策略。"""
    def __init__(self,seed,edge=False):
        rng=random.Random(seed)
        self.seed=seed
        self.targets={}
        for ch in rng.sample(range(1,21),rng.randint(10,16)):
            r=1800 if edge else 1800*math.sqrt(rng.random())
            t=rng.random()*2*math.pi
            self.targets[ch]=((r*math.cos(t),r*math.sin(t)),1000 if edge else rng.uniform(1000,1500))
        self.total=len(self.targets)
        self.done=set(); self.q=(0,0); self.ch=1; self.vt=0; self.cache={}
    def post(self,path,payload,timeout):
        key=payload['request_id']
        if key in self.cache:
            return self.cache[key]
        result={'accepted':True,'real_timestamp_ms':int(time.time()*1000)}
        if path=='/enter':
            result.update(remaining_real_duration_s=1200,max_virtual_duration_s=360000,max_real_duration_s=1200)
        elif path in ('/measure','/clear'):
            q=(payload['position']['x'],payload['position']['y']); ch=payload['channel']
            self.vt+=dist(self.q,q)/5; self.q=q
            target=self.targets.get(ch)
            d=dist(q,target[0]) if target and ch not in self.done else math.inf
            if path=='/measure':
                self.vt+=5+(ch!=self.ch); self.ch=ch
                kind='no_signal' if not target or d>target[1] else ('near' if d<=5 else 'direction')
                result['measure_result']=kind
                if kind=='direction':
                    token=f'{self.seed}:{ch}:{q[0]:.9f}:{q[1]:.9f}'.encode()
                    error=2*int.from_bytes(hashlib.sha256(token).digest()[:8],'big')/(2**64-1)-1
                    angle=math.degrees(math.atan2(target[0][1]-q[1],target[0][0]-q[0]))
                    result['svd_deg']=round((angle+error)%360,2)%360
            else:
                success=d<=20; self.vt+=5 if success else 3
                result['clear_result']='success' if success else 'no_target_in_range'
                if success: self.done.add(ch)
        else:
            result['exit_reason']='user_exit'
        result['virtual_time_s']=self.vt
        self.cache[key]=result
        return result

class BudgetStop(RuntimeError):
    pass

class Robot:
    def __init__(self,backend,robot_id,out=None,quiet=False,layout=None):
        self.backend=backend; self.robot_id=robot_id; self.out=out; self.quiet=quiet
        self.layout=layout or resolve_layout()
        self.prefix=uuid.uuid4().hex[:12]; self.seq=0; self.q=(0.,0.); self.ch=1
        self.vt=0.; self.vmax=360000.; self.deadline=math.inf; self.started=time.monotonic()
        self.done=set(); self.absent=set(); self.searched={c:set() for c in range(1,21)}
        self.distance=0.; self.measures=0; self.switches=0; self.clears=0; self.fallbacks=0; self.information_records=[]
        self.problem1_geometry_records=[]
        self.log=None
        if out:
            out.mkdir(parents=True,exist_ok=True)
            self.log=(out/'actions.jsonl').open('w',encoding='utf-8')
    def write_log(self,record):
        if self.log:
            self.log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); self.log.flush()
    def call(self,path,q=None,ch=None):
        if path in ('/measure','/clear'):
            cost=dist(self.q,q)/5+6
            if time.monotonic()>self.deadline-12 or self.vt+cost>self.vmax-1:
                raise BudgetStop('接近运行时限，任务未完成，保存当前结果并退出。')
        self.seq+=1
        payload={'arena_id':'default','robot_id':self.robot_id,'request_id':f'{self.prefix}-{self.seq}'}
        if q is not None:
            payload.update(position={'x':float(q[0]),'y':float(q[1])},channel=ch)
        self.write_log({'event':'request','path':path,'payload':payload})
        for attempt in range(3):
            if time.monotonic()>=self.deadline:
                raise RuntimeError('本局已到现实截止时间；不再发送请求。')
            try:
                result=self.backend.post(path,payload,max(0.1,min(4.,self.deadline-time.monotonic())))
                break
            except HTTPError as exc:
                self.write_log({'event':'http_error','code':exc.code,'request_id':payload['request_id']})
                raise RuntimeError(f'HTTP {exc.code}，请检查模拟器状态及请求参数。') from exc
            except (URLError,TimeoutError,ConnectionError,OSError,json.JSONDecodeError) as exc:
                self.write_log({'event':'network_error','attempt':attempt+1,'detail':str(exc),'request_id':payload['request_id']})
                if attempt==2:
                    raise RuntimeError('连接失败或响应丢失，动作结果可能未知；已停止，不发送后续动作。请检查模拟器界面和日志。') from exc
                time.sleep(0.3) # 同一个payload/request_id重试，绝不并发
        self.write_log({'event':'response','path':path,'request_id':payload['request_id'],'response':result})
        if result.get('accepted') is not True:
            # The protocol sets virtual_time_s to 0 for accepted=false.  That
            # value is not the current clock and must not overwrite the last
            # accepted timestamp before the request is rejected.
            raise RuntimeError('请求被拒绝 accepted=false；确认参赛队号、接口就绪状态。')
        try:
            reported_vt=float(result['virtual_time_s'])
        except (KeyError,TypeError,ValueError,OverflowError) as exc:
            raise RuntimeError('响应缺少有限 virtual_time_s。') from exc
        if not math.isfinite(reported_vt) or reported_vt<0.:
            raise RuntimeError('响应 virtual_time_s 非法（应为非负有限数）。')
        enter_remaining = enter_vmax = None
        if path=='/enter':
            # The simulator contract exposes the real-time budget as a
            # bounded numeric value (0--1200 seconds).  Validate it before
            # changing any local clock/deadline state; malformed NaN/Inf or
            # negative values must stop the run instead of disabling the
            # safety budget through float propagation.
            try:
                enter_remaining=float(result['remaining_real_duration_s'])
            except (KeyError,TypeError,ValueError,OverflowError) as exc:
                raise RuntimeError('响应缺少有限 remaining_real_duration_s。') from exc
            if (not math.isfinite(enter_remaining) or
                    enter_remaining<0. or enter_remaining>1200. or
                    enter_remaining!=math.floor(enter_remaining)):
                raise RuntimeError('响应 remaining_real_duration_s 非法（应为[0,1200]内的有限整数）。')
            try:
                enter_vmax=float(result.get('max_virtual_duration_s',360000))
            except (TypeError,ValueError,OverflowError) as exc:
                raise RuntimeError('响应缺少有限 max_virtual_duration_s。') from exc
            if not math.isfinite(enter_vmax) or enter_vmax<=0.:
                raise RuntimeError('响应 max_virtual_duration_s 非法（应为有限正数）。')
        measure_result = clear_result = None
        if path=='/measure':
            measure_result=result.get('measure_result')
            if measure_result not in ('near','direction','no_signal'):
                raise RuntimeError('未知检测响应')
            if measure_result=='direction':
                try:
                    svd=float(result['svd_deg'])
                except (KeyError,TypeError,ValueError,OverflowError) as exc:
                    raise RuntimeError('direction 响应缺少有限 svd_deg。') from exc
                if not math.isfinite(svd) or not 0.<=svd<360.:
                    raise RuntimeError('direction 响应 svd_deg 非法（应位于[0,360)）。')
        elif path=='/clear':
            clear_result=result.get('clear_result')
            if clear_result not in ('success','no_target_in_range'):
                raise RuntimeError('未知清除响应')
        if path in ('/measure','/clear'):
            # Accepted movement/actions must advance (or, for a zero-length
            # action, preserve) the simulator clock.  A backwards timestamp
            # indicates a protocol/transport inconsistency and must not be
            # allowed to corrupt the accounting used for subsequent budgets.
            if reported_vt+1e-9<self.vt:
                raise RuntimeError('响应 virtual_time_s 回退。')
            self.vt=reported_vt
        elif path=='/enter':
            self.vt=reported_vt
        else:
            # /exit does not advance virtual time.  Preserve the latest
            # accepted action clock if a malformed/legacy endpoint returns 0;
            # accept a finite equal-or-larger timestamp from a conforming one.
            self.vt=max(self.vt,reported_vt)
        if path=='/enter':
            # Values were validated above, before mutating the local budget.
            self.deadline=time.monotonic()+enter_remaining
            self.vmax=enter_vmax
        if q is not None:
            self.distance+=dist(self.q,q); self.q=tuple(q)
        if path=='/measure':
            self.measures+=1; self.switches+=ch!=self.ch; self.ch=ch
        if path=='/clear':
            self.clears+=1
            if clear_result=='success': self.done.add(ch)
        if not self.quiet and q is not None:
            kind=result.get('measure_result',result.get('clear_result'))
            print(f'{self.seq:4d} {path:8s} ch={ch:2d} ({q[0]:8.1f},{q[1]:8.1f}) {kind:20s} T={self.vt:.1f}s cleared={len(self.done)}',flush=True)
        return result
    def clear(self,ch,q):
        return self.call('/clear',q,ch)['clear_result']=='success'
    def record_failed_clear(self,ch,target,q,source):
        """Apply failed-clear information and invalidate any stale plan."""
        info=condition_failed_clear(target,q)
        self.beliefs[ch].update(mean=target['mean'],covariance=target['covariance'])
        info={'source':source,'failed_position':list(map(float,q)),
              'remaining_fallback_candidates':(
                  len(target['fallback']) if target.get('fallback') is not None else None),
              **info}
        self.write_log({'event':'failed_clear_update','channel':ch,**info})
        return info
    def condition_discovered_negative(self,ch,q):
        """Condition a known target on a negative receive observation.

        A later no-signal response cannot revoke discovery.  Keep the target's
        geometric outer region unchanged, update only the fixed-radius
        planning posterior, invalidate any planned measurement, and preserve
        existence probability one.
        """
        target=self.targets[ch]
        old_negatives=list(target.get('negatives',[]))
        target.setdefault('negatives',[]).append(tuple(q))
        target['planned']=None
        posterior=update_gaussian_negative(target,old_negatives)
        # The cache key is the station count; a negative receive observation
        # changes the weights without changing that count.
        target.pop('probe_base',None)
        belief=self.beliefs[ch]
        belief.update(mean=target['mean'],covariance=target['covariance'])
        belief['existence']=1.
        return posterior
    def clear_point(self, poly, center):
        """在20米共同覆盖圆盘交集中选离当前位置近的清除点。"""
        try:
            vertices=normalize_points(poly)
        except (TypeError,ValueError):
            vertices=np.empty((0,2),dtype=float)
        if not len(vertices):
            # No finite clear point can certify an empty region.  The caller
            # will treat this as an exhausted fallback instead of crashing in
            # min/max or SLSQP setup.
            return tuple(_finite_point(center))
        current=_finite_point(self.q); scale=1000.
        try:
            supplied=np.asarray(center,dtype=float)
        except (TypeError,ValueError):
            supplied=np.empty(0,dtype=float)
        if supplied.shape==(2,) and np.all(np.isfinite(supplied)):
            fallback=supplied
        else:
            fallback=np.asarray(mec(vertices.tolist())[0],dtype=float)
        def safe(z):
            try:
                value=np.asarray(z,dtype=float)
                if value.shape!=(2,) or not np.all(np.isfinite(value)):
                    return np.full(len(vertices),-math.inf)
                return (19.8**2-np.sum((value*scale-vertices)**2,axis=1))/400.
            except (TypeError,ValueError):
                return np.full(len(vertices),-math.inf)
        try:
            result=minimize(lambda z:float(np.sum((np.asarray(z,dtype=float)-current/scale)**2)),
                            fallback/scale,method='SLSQP',constraints=[{'type':'ineq','fun':safe}],
                            options={'maxiter':40,'ftol':1e-9})
        except Exception:
            return tuple(float(v) for v in fallback)
        candidate=getattr(result,'x',None)
        try:
            candidate=np.asarray(candidate,dtype=float)
            if candidate.shape!=(2,) or not np.all(np.isfinite(candidate)):
                raise ValueError
            q=tuple(float(v) for v in (candidate*scale))
        except (TypeError,ValueError):
            return tuple(float(v) for v in fallback)
        try:
            if (getattr(result,'success',False) and
                    max(dist(q,p) for p in vertices)<=19.80001):
                return q
        except (TypeError,ValueError):
            pass
        # An unsuccessful or non-finite optimizer result must never be used as
        # a clear point.  The MEC center is the deterministic safe fallback.
        return tuple(float(v) for v in fallback)

    def make_fallback(self, target):
        """对当前保守凸区域作25米网格覆盖；失败后依次尝试剩余格。"""
        poly=target['poly']; cells=[]
        if not poly:
            target['fallback']=[]
            self.fallbacks+=1
            return
        xmin=min(p[0] for p in poly); xmax=max(p[0] for p in poly)
        ymin=min(p[1] for p in poly); ymax=max(p[1] for p in poly)
        for iy in range(max(1,math.ceil((ymax-ymin)/25))):
            y0=ymin+25*iy; y1=min(ymax,y0+25); row=[]
            for ix in range(max(1,math.ceil((xmax-xmin)/25))):
                x0=xmin+25*ix; x1=min(xmax,x0+25); part=poly
                for a,b in [((1,0),x1),((-1,0),-x0),((0,1),y1),((0,-1),-y0)]:
                    part=clip(part,a,b)
                if part and not any(max(dist(v,failed) for v in part)<=20 for failed in target.get('failed_clears',[])):
                    row.append(((x0+x1)/2,(y0+y1)/2))
            cells.extend(row if iy%2==0 else row[::-1])
        if cells and dist(self.q,cells[-1])<dist(self.q,cells[0]): cells.reverse()
        target['fallback']=cells; self.fallbacks+=1

    def schedule_run(self):
        self.targets={}
        self.beliefs={ch:{'mean':np.zeros(2),'covariance':np.eye(2)*1800**2/4,
                          'existence':BAYES_CONFIG['prior_exists'],'negatives':[]} for ch in range(1,21)}
        self.coverage_map=CoverageMap(SCHEDULE_CONFIG['cell_size']) # 仅概率求积辅助，不作为完成证书
        scans=scan_points(self.layout['ring_points'],self.layout['ring_radius'])
        certificate=self.layout['certificate']
        self.cpp_plan={'layout_name':self.layout['layout_name'],
                       'ring_points':self.layout['ring_points'],
                       'ring_radius':self.layout['ring_radius'],
                       'waypoint_count':len(scans),'waypoints':scans,
                       'certificate':certificate}
        if self.out:(self.out/'scan_plan.json').write_text(json.dumps(self.cpp_plan,indent=2),encoding='utf-8')
        self.write_log({'event':'coverage_certificate',**self.cpp_plan})
        self.scheduler_counts={'search_measures':0,'localization_measures':0,'certified_clears':0,
            'fallback_clears':0,'decisions':0,'probe_attempts':0,'probe_successes':0,
            'joint_sequences_planned':0,'on_route_probes':0,'exact_routes':0,'approximate_routes':0,
            'batch_scan_visits':0,'batch_scan_channels':0,
            'route_aligned_plans':0,'free_measurement_plans':0,
            'dop_tie_break_decisions':0,'dop_tie_break_overrides':0}
        cache_key=None;cache_value=None
        while len(self.done)<16 and len(self.done|self.absent)<20:
            if time.monotonic()>self.deadline-20:raise BudgetStop('滚动规划接近现实截止时间。')
            unknown=[ch for ch in range(1,21) if ch not in self.targets and ch not in self.absent]
            nodes=[]
            for i,q in enumerate(scans):
                channels=[ch for ch in unknown if i not in self.searched[ch]]
                if channels:nodes.append({'type':'scan','id':f'scan-{i}','index':i,'position':q,'channels':channels})
            for ch,target in self.targets.items():
                if ch not in self.done:
                    nodes.append({'type':'source','id':f'source-{ch}','channel':ch,
                                  'position':tuple(target.get('mean',target['center']))})
            if not nodes:raise RuntimeError('无任务节点，但未能证明全部清除。')
            key=(tuple(self.q),tuple((n['id'],tuple(n['position'])) for n in nodes))
            if key!=cache_key:
                cache_value=rolling_open_route(self.q,nodes);cache_key=key
            order,route_info=cache_value
            self.scheduler_counts['exact_routes' if route_info['exact'] else 'approximate_routes']+=1
            node=nodes[order[0]]
            next_stop=nodes[order[1]]['position'] if len(order)>1 else None
            self.scheduler_counts['decisions']+=1
            self.write_log({'event':'rolling_route','nodes':nodes,'order':[nodes[i]['id'] for i in order],**route_info})
            if node['type']=='scan':
                q=node['position']
                # 到达扫描点后批量检测本节点全部尚未搜索频道。当前频道优先，
                # 其余频道按环形顺序检测，避免同一扫描点被反复加入滚动路线。
                pending=list(node['channels'])
                # All unknown channels remain scheduled, but the configured
                # score decides the order in this batch.  Keep the current
                # channel first afterward to avoid an unnecessary switch.
                search_scores={}
                for pending_ch in pending:
                    information,probability=detection_information(
                        self.beliefs[pending_ch],self.coverage_map,q)
                    search_scores[pending_ch]=information+SCHEDULE_CONFIG['search_gain']*probability
                pending.sort(key=lambda pending_ch:(-search_scores[pending_ch],pending_ch))
                # V2：把当前扫描点同时当作已发现目标的顺路补测点。只有当整个
                # 定位多边形都位于1000米接收圆内，且预计信息量达标时才加入。
                route_targets=[]
                for target_ch,target in self.targets.items():
                    if target_ch in self.done:continue
                    score=mandatory_stop_measurement_score(target,q)
                    if score is not None:route_targets.append((score['information_nats'],target_ch,score))
                route_targets.sort(reverse=True)
                route_targets=route_targets[:ROUTE_MEASURE_CONFIG['max_target_channels_per_stop']]
                pending.extend(ch for _,ch,_ in route_targets if ch not in pending)
                if self.ch in pending:
                    start=pending.index(self.ch)
                    pending=pending[start:]+pending[:start]
                self.scheduler_counts['batch_scan_visits']+=1
                self.scheduler_counts['batch_scan_channels']+=len(pending)
                self.write_log({'event':'batch_scan_start','scan_id':node['id'],'index':node['index'],
                                'position':q,'channels':pending,'start_channel':self.ch,
                                'search_scores':{str(ch):search_scores[ch] for ch in search_scores},
                                'route_aligned_targets':[{'channel':ch,**score} for _,ch,score in route_targets]})
                measured=[]
                for ch in pending:
                    # 前面的测量可能已将频道清除或认证为不存在；这种情况下跳过。
                    if ch in self.done or ch in self.absent or node['index'] in self.searched[ch]:
                        continue
                    was_target=ch in self.targets
                    res=self.call('/measure',q,ch)
                    if was_target:
                        self.scheduler_counts['route_aligned_plans']+=1
                        self.scheduler_counts['localization_measures']+=1
                    else:
                        self.scheduler_counts['search_measures']+=1
                    measured.append(ch)
                    self.searched[ch].add(node['index'])
                    if res['measure_result']=='no_signal':
                        self.coverage_map.update_negative(ch,q)
                        self.beliefs[ch]['negatives'].append(q)
                        # 仅尚未发现的频道可由全覆盖扫描认证为不存在；已发现目标的
                        # no_signal 只作为固定接收半径的负观测，不得误判为无源。
                        if ch in self.targets:
                            belief=self.condition_discovered_negative(ch,q)
                            self.write_log({'event':'negative_update','channel':ch,
                                            'known_target':True,**belief,
                                            'existence_probability':self.beliefs[ch]['existence']})
                        elif len(self.searched[ch])==len(scans):
                            belief=update_unknown_belief(self.beliefs[ch],self.coverage_map,ch)
                            self.absent.add(ch);self.beliefs[ch]['existence']=0.
                            # The certificate changes the final state from a
                            # probabilistic estimate to a proved absence; log
                            # the post-certificate value rather than the
                            # intermediate posterior.
                            belief['existence_probability']=0.
                            belief['certified_absent']=True
                            self.write_log({'event':'negative_update','channel':ch,**belief})
                        else:
                            belief=update_unknown_belief(self.beliefs[ch],self.coverage_map,ch)
                            self.write_log({'event':'negative_update','channel':ch,**belief})
                    else:
                        self.accept_bearing(ch,q,res)
                self.write_log({'event':'batch_scan_end','scan_id':node['id'],'index':node['index'],
                                'position':q,'measured_channels':measured,
                                'new_current_channel':self.ch,'done':sorted(self.done),
                                'absent':sorted(self.absent)})
                # 批次结束后再统一重建源节点与剩余扫描节点并规划下一段路径。
                cache_key=None
                continue
            ch=node['channel'];target=self.targets[ch]
            if target.get('near') is not None or target['radius']<=19.8:
                q=target.get('near')
                if q is None:q=self.clear_point(target['poly'],target['center'])
                self.write_log({'event':'local_decision','channel':ch,'choice':'certified_clear'})
                if not self.clear(ch,q):raise RuntimeError('可靠清除与响应矛盾')
                self.scheduler_counts['certified_clears']+=1;continue
            if target.get('fallback') is not None:
                if not target['fallback']:raise RuntimeError('网格兜底耗尽而目标未清除')
                q=target['fallback'].pop(0)
                success=self.clear(ch,q)
                if not success:self.record_failed_clear(ch,target,q,'fallback')
                self.scheduler_counts['fallback_clears']+=1;continue
            if target.get('planned') is None:
                if len(target['stations'])>=8:self.make_fallback(target);continue
                free_q,free_detail=bayes_select(target['poly'],target['stations'],self.q,target,self.deadline)
                dop_detail=free_detail.get('problem2_dop_tiebreak')
                if dop_detail is not None:
                    self.scheduler_counts['dop_tie_break_decisions']+=1
                    self.scheduler_counts['dop_tie_break_overrides']+=int(dop_detail['override'])
                    self.write_log({'event':'problem2_dop_tiebreak','channel':ch,**dop_detail})
                if free_q is None:
                    detail=free_detail;q=None
                else:
                    q,detail=route_aligned_measurement(
                        target['poly'],target['stations'],self.q,target,next_stop,free_q,free_detail)
                    key_name='route_aligned_plans' if detail.get('route_aligned') else 'free_measurement_plans'
                    self.scheduler_counts[key_name]+=1
                self.information_records.append(detail)
                self.write_log({'event':'measurement_plan','channel':ch,**detail})
                if q is None:self.make_fallback(target);continue
                target['planned']=q
            q=target['planned']
            measure_cost=dist(self.q,q)/5+5+(self.ch!=ch)+dist(q,target['center'])/5+5
            if next_stop is not None:measure_cost+=dist(target['center'],next_stop)/5
            probe=joint_probe_plan(target,self.q,self.ch,ch,q,True,next_stop)
            self.write_log({'event':'local_decision','channel':ch,
                'expected_measure_then_clear_s':measure_cost,
                'expected_probe_sequence_s':probe['expected_time_s'] if probe else None,
                'probe_plan':probe,'choice':'probe' if probe else 'measure',
                'note':'补测一次后清除为预测代理；成功仍需收到clear成功响应。'})
            if probe:
                target['probe_attempts']=target.get('probe_attempts',0)+1
                self.scheduler_counts['probe_attempts']+=1
                self.scheduler_counts['joint_sequences_planned']+=int(len(probe['sequence'])>=2)
                self.scheduler_counts['on_route_probes']+=int(probe['worst_branch_extra_distance_m']<1e-5)
                success=self.clear(ch,probe['position'])
                if success:self.scheduler_counts['probe_successes']+=1
                else:self.record_failed_clear(ch,target,probe['position'],'probe')
                self.write_log({'event':'probe_result','channel':ch,'success':success,'plan':probe})
            else:
                res=self.call('/measure',q,ch);self.scheduler_counts['localization_measures']+=1
                if res['measure_result']=='no_signal':raise RuntimeError('接收认证点返回无信号，请确认问题3全向案例。')
                self.accept_bearing(ch,q,res)
        if len(self.done)<10:raise RuntimeError('清除数低于题设下界10，未确认完成。')

    def accept_bearing(self,ch,q,result):
        target=self.targets.setdefault(ch,{'poly':initial_poly(),'stations':[],'planned':None,'fallback':None,
                                  'negatives':list(self.beliefs[ch]['negatives'])})
        self.beliefs[ch]['existence']=1.;target['planned']=None
        if result['measure_result']=='near':
            target.update(near=q,center=q,radius=5.,mean=np.asarray(q),covariance=np.eye(2)*25/4)
            self.beliefs[ch].update(mean=target['mean'],covariance=target['covariance'])
            if not self.clear(ch,q):raise RuntimeError('near后清除失败')
            self.scheduler_counts['certified_clears']+=1
        else:
            target['angle']=result['svd_deg'];target['poly']=update_poly(target['poly'],q,target['angle'])
            target['stations'].append(q);target['center'],target['radius']=mec(target['poly'])
            geometry=problem1_geometry_certificate(target['poly'],target['center'],target['radius'])
            self.problem1_geometry_records.append(geometry)
            posterior=update_gaussian(target,q,target['angle'])
            self.beliefs[ch].update(mean=target['mean'],covariance=target['covariance'])
            self.write_log({'event':'gaussian_update','channel':ch,**posterior,
                            'problem1_geometry':geometry})

    def run(self):
        status='incomplete'; error=None; entered=False
        try:
            self.call('/enter')
            # Program runtime is measured from successful entry, not from
            # object construction while the simulator is still being opened.
            self.started=time.monotonic(); entered=True
            self.schedule_run()
            status='complete'
            self.call('/exit')
        except BudgetStop as exc:
            status='budget_stop'; error=str(exc)
            if entered and time.monotonic()<self.deadline:
                try: self.call('/exit')
                except Exception as exit_exc: error+='；退出失败：'+str(exit_exc)
        except (Exception,KeyboardInterrupt) as exc:
            status='error'; error=str(exc) or '用户中断'
            # 通信结果可能未知，绝不在错误后继续发送新动作或探测/exit。
        geometry_records=[g for g in self.problem1_geometry_records
                          if math.isfinite(g['jung_ratio']) and math.isfinite(g['radius_to_diameter'])]
        geometry_summary={'sample_count':len(self.problem1_geometry_records),
            'finite_sample_count':len(geometry_records),
            'max_jung_ratio':max((g['jung_ratio'] for g in geometry_records),default=None),
            'mean_radius_to_diameter':(sum(g['radius_to_diameter'] for g in geometry_records)/len(geometry_records)
                                       if geometry_records else None),
            'jung_violation_count':sum(not g['jung_certified'] for g in self.problem1_geometry_records)}
        summary={'status':status,'error':error,'cleared_count':len(self.done),
                 'cleared_channels':sorted(self.done),'absent_channels':sorted(self.absent),
                 'virtual_time_s':self.vt,'average_clear_time_s':self.vt/len(self.done) if self.done else None,
                 'program_runtime_s':time.monotonic()-self.started,'distance_m':self.distance,
                 'measure_count':self.measures,'switch_count':self.switches,'clear_attempts':self.clears,
                 'failed_clear_attempts':self.clears-len(self.done),'fallback_count':self.fallbacks,
                 'information_selections':len(self.information_records),
                 'strategy':'parameterized_ring_rolling_open_route',
                 'layout_name':self.layout['layout_name'],
                 'ring_points':self.layout['ring_points'],'ring_radius':self.layout['ring_radius'],
                 'joint_config':dict(JOINT_CONFIG),'probe_config':dict(PROBE_CONFIG),
                 'bayes_config':dict(BAYES_CONFIG),'q12_link_config':dict(Q12_LINK_CONFIG),
                 'problem1_geometry':geometry_summary,
                 'scheduler':getattr(self,'scheduler_counts',{}),'scheduler_config':dict(SCHEDULE_CONFIG),
                 'clear_ratio':None,'note':'真实总数仅演练结束后可见；clear_ratio需用实际总数计算。'}
        expected=self.distance/5+self.switches+5*self.measures+3*self.clears+2*len(self.done)
        summary['accounting_delta_s']=self.vt-expected
        if abs(summary['accounting_delta_s'])<1e-9:summary['accounting_delta_s']=0.
        if self.out:
            if hasattr(self,'beliefs'):
                posteriors={str(ch):{'mean':b['mean'].tolist(),'covariance':b['covariance'].tolist(),
                                     'existence_probability':b['existence'],
                                     'status':'cleared' if ch in self.done else ('absent' if ch in self.absent else ('detected' if ch in self.targets else 'unknown'))}
                            for ch,b in self.beliefs.items()}
                (self.out/'gaussian_posteriors.json').write_text(json.dumps(posteriors,ensure_ascii=False,indent=2),encoding='utf-8')
            if hasattr(self,'coverage_map'):
                fm=self.coverage_map
                snapshot={'cell_size_m':fm.step,'centers':fm.centers.tolist(),
                          'remaining_cell_indices':{str(ch):np.flatnonzero(fm.remaining[ch-1]).tolist()
                                                    for ch in range(1,21) if ch not in self.targets},
                          'absent_channels':sorted(self.absent), 'cleared_channels':sorted(self.done),
                          'note':'方格外包连续位置；未发现频道仅凭整格无信号认证删除。'}
                (self.out/'coverage_map.json').write_text(json.dumps(snapshot,ensure_ascii=False),encoding='utf-8')
            (self.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
            with (self.out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
        if self.log: self.log.close()
        return summary

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)

def scenario_record(backend):
    return {'seed':backend.seed,'total':backend.total,
            'sources':[{'channel':ch,'position':list(value[0]),
                        'reception_radius_m':value[1]}
                       for ch,value in sorted(backend.targets.items())]}

def run_local_case(seed, case_number, layout, case_dir):
    backend=LocalBackend(seed,edge=False)
    truth=scenario_record(backend)
    robot=Robot(backend,'LOCAL-TEST',case_dir,quiet=True,layout=layout)
    result=robot.run()
    result.update(true_total=backend.total,
                  clear_ratio=len(backend.done)/backend.total,
                  seed=seed,case=case_number)
    # Robot.run先写基础summary；补充仅本地评估器可知的真值指标。
    (case_dir/'summary.json').write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    write_csv(case_dir/'summary.csv',[result])
    (case_dir/'scenario.json').write_text(
        json.dumps(truth,ensure_ascii=False,indent=2),encoding='utf-8')
    success=(result['status']=='complete' and len(backend.done)==backend.total and
             abs(result['accounting_delta_s'])<1e-5 and
             all(len(robot.searched[ch])==layout['ring_points']+1
                 for ch in result['absent_channels']))
    row={'layout_name':layout['layout_name'],'ring_points':layout['ring_points'],
         'ring_radius':layout['ring_radius'],'case':case_number,'seed':seed,
         'true_total':backend.total,'total':backend.total,
         'cleared_count':len(backend.done),'cleared':len(backend.done),
         'success':success,'status':result['status'],'clear_ratio':result['clear_ratio'],
         'virtual_time_s':result['virtual_time_s'],
         'average_clear_time_s':result['average_clear_time_s'],
         'distance_m':result['distance_m'],'measure_count':result['measure_count'],
         'failed_clear_attempts':result['failed_clear_attempts'],
         'program_runtime_s':result['program_runtime_s'],
         'probe_attempts':result['scheduler'].get('probe_attempts',0),
         'probe_successes':result['scheduler'].get('probe_successes',0),
         'exact_routes':result['scheduler'].get('exact_routes',0),
         'approximate_routes':result['scheduler'].get('approximate_routes',0),
         'error':result['error']}
    return row,truth

def summarize_layout(layout, rows):
    def mean(key):return sum(float(r[key]) for r in rows)/len(rows)
    total_times=[float(r['virtual_time_s']) for r in rows]
    avg_time=sum(total_times)/len(total_times)
    variance=sum((x-avg_time)**2 for x in total_times)/len(total_times)
    total_sources=sum(int(r['true_total']) for r in rows)
    return {'layout_name':layout['layout_name'],'ring_points':layout['ring_points'],
            'ring_radius':layout['ring_radius'],'case_count':len(rows),
            'success_count':sum(bool(r['success']) for r in rows),
            'mean_clear_ratio':mean('clear_ratio'),
            'mean_virtual_time_s':avg_time,
            'std_virtual_time_s':math.sqrt(variance),
            'pooled_time_per_source_s':sum(total_times)/total_sources,
            'mean_distance_m':mean('distance_m'),
            'mean_measure_count':mean('measure_count'),
            'mean_program_runtime_s':mean('program_runtime_s'),
            'failed_clear_attempts':sum(int(r['failed_clear_attempts']) for r in rows)}

def self_test(count,seed_start=2026091200,output_dir=None,layout=None):
    """运行单布局本地案例；目录为 root/layout_name/case_NN。"""
    root=Path(output_dir) if output_dir is not None else DEFAULT_LOCAL_TEST_OUTPUT
    root.mkdir(parents=True,exist_ok=True)
    layout=layout or resolve_layout();seeds=list(range(seed_start,seed_start+count))
    rows=[]
    for i,seed in enumerate(seeds,1):
        case=root/layout['layout_name']/f'case_{i:02d}'
        row,_=run_local_case(seed,i,layout,case)
        rows.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    summary=summarize_layout(layout,rows)
    report={'note':'本地随机合成环境，非官方演练；仿真真值只供评估器使用',
            'layout':layout,'seeds':seeds,'cases':rows,'summary':summary,
            'mean_total_time_s':summary['mean_virtual_time_s'],
            'pooled_time_per_source_s':summary['pooled_time_per_source_s']}
    (root/'test_results.json').write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return report

def compare_layouts(count,seed_start=2026091200,output_dir=None):
    """以共同随机数对全部预设布局作配对仿真并保存完整过程。"""
    root=Path(output_dir) if output_dir is not None else OUTPUT_ROOT / 'layout_comparison'
    root.mkdir(parents=True,exist_ok=True)
    seeds=list(range(seed_start,seed_start+count));all_rows=[];summaries=[]
    scenario_fingerprints={}
    resolved=[resolve_layout(name) for name in LAYOUT_PRESETS]
    for layout in resolved:
        rows=[]
        for i,seed in enumerate(seeds,1):
            case=root/layout['layout_name']/f'case_{i:02d}'
            row,truth=run_local_case(seed,i,layout,case)
            fingerprint=hashlib.sha256(json.dumps(truth,sort_keys=True).encode()).hexdigest()
            old=scenario_fingerprints.setdefault(seed,fingerprint)
            if old!=fingerprint:
                raise RuntimeError(f'seed {seed} 在不同布局生成了不同真值')
            rows.append(row);all_rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
        summaries.append(summarize_layout(layout,rows))
    write_csv(root/'comparison_cases.csv',all_rows)
    write_csv(root/'comparison_summary.csv',summaries)
    results={'note':'所有布局共享相同seed与LocalBackend生成规则，可进行配对比较',
             'seeds':seeds,'cases':all_rows,'summaries':summaries}
    (root/'comparison_results.json').write_text(
        json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={'experiment':'q3_multi_layout_common_random_numbers',
              'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'description':'问题三多布局本地合成仿真；各布局使用完全相同的seed和目标真值。',
              'case_count':count,'seed_start':seed_start,'seeds':seeds,
              'presets':LAYOUT_PRESETS,
              'coverage_certificates':{x['layout_name']:x['certificate'] for x in resolved},
              'output_structure':'<layout_name>/case_NN/保存每局完整过程文件'}
    (root/'experiment_manifest.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summaries':summaries},ensure_ascii=False,indent=2))
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--search-gain',type=float,default=0.2,help='扫描批次收益权重，默认0.2，非存在概率')
    parser.add_argument('--cell-size',type=float,default=100.,help='保守方格边长，默认100米，50到200米')
    parser.add_argument('--lookahead',type=float,default=1.0,help='插入任务的后续绕行权重，默认1.0')
    parser.add_argument('--quad-depth',type=int,default=3,help='后验积分三角形细分层数，1到4')
    parser.add_argument('--noise-deg',type=float,default=1/math.sqrt(3),help='近似测角标准差，默认0.577度；不改变硬误差界')
    parser.add_argument('--prior-exists',type=float,default=0.65,help='每频道初始存在概率近似，默认0.65')
    parser.add_argument('--clear-detour',type=float,default=5.,help='失败分支相对既定路线最多增加的米数；0为严格顺路')
    parser.add_argument('--clear-horizon',type=int,default=3,help='联合规划深度，1到3')
    parser.add_argument('--min-clear-saving',type=float,default=0.,help='预计至少节省的秒数')
    parser.add_argument('--max-probes',type=int,default=5,help='每目标试探clear次数上限；0为禁用')
    parser.add_argument('--min-probe-prob',type=float,default=0.2,help='试探成功概率估计下限')
    parser.add_argument('--probe-radius',type=float,default=80.,help='允许试探的定位覆盖圆半径上限，米')
    parser.add_argument('--layout',choices=list(LAYOUT_PRESETS),default=DEFAULT_LAYOUT,
                        help='预设布局，默认 ring8_939')
    parser.add_argument('--ring-points',type=int,
                        help='自定义外围点数，覆盖所选预设的 n')
    parser.add_argument('--ring-radius',type=float,
                        help='自定义布局半径（米），覆盖所选预设的 R')
    parser.add_argument('--compare-layouts',action='store_true',
                        help='对全部预设布局使用相同seed批量比较（需同时指定--self-test N）')
    parser.add_argument('--robot-id',help='与模拟器登录参赛队号完全一致')
    parser.add_argument('--base-url',default='http://127.0.0.1:2026')
    parser.add_argument('--output',help='日志目录；默认按时间创建')
    parser.add_argument('--seed-start',type=int,default=2026091200,help='本地测试起始随机种子')
    parser.add_argument('--test-output',help='本地测试结果目录；默认 outputs/t3/local_tests')
    parser.add_argument('--self-test',type=int,metavar='N',help='运行N个本地随机案例，不连接官方模拟器')
    args=parser.parse_args()
    if not (0<args.noise_deg<=5 and 0<args.prior_exists<1 and 1<=args.quad_depth<=4):
        parser.error('0<noise-deg<=5，0<prior-exists<1，1<=quad-depth<=4')
    if not (0<args.search_gain<=5 and 0<=args.lookahead<=1 and 50<=args.cell_size<=200):
        parser.error('0<search-gain<=5，0<=lookahead<=1，50<=cell-size<=200')
    SCHEDULE_CONFIG.update(search_gain=args.search_gain,lookahead=args.lookahead,cell_size=args.cell_size)
    if not (0<=args.max_probes<=20 and 0<args.min_probe_prob<1 and 20<=args.probe_radius<=200):
        parser.error('0<=max-probes<=20，0<min-probe-prob<1，20<=probe-radius<=200')
    if not (0<=args.clear_detour<=50 and 1<=args.clear_horizon<=3 and args.min_clear_saving>=0):
        parser.error('0<=clear-detour<=50，1<=clear-horizon<=3，min-clear-saving>=0')
    JOINT_CONFIG.update(max_detour_m=args.clear_detour,horizon=args.clear_horizon,min_saving_s=args.min_clear_saving)
    PROBE_CONFIG.update(max_attempts=args.max_probes,min_probability=args.min_probe_prob,max_radius=args.probe_radius)
    BAYES_CONFIG.update(noise_deg=args.noise_deg,prior_exists=args.prior_exists,quad_depth=args.quad_depth)
    if args.compare_layouts and args.self_test is None:
        parser.error('--compare-layouts 必须与 --self-test N 同时使用')
    if args.compare_layouts and (args.ring_points is not None or args.ring_radius is not None):
        parser.error('--compare-layouts 比较固定预设，不接受 --ring-points/--ring-radius')
    try:
        layout=resolve_layout(args.layout,args.ring_points,args.ring_radius)
    except (ValueError,KeyError) as exc:
        parser.error(str(exc))
    if args.self_test is not None:
        if args.self_test<1: parser.error('N必须大于0')
        if args.compare_layouts:
            report=compare_layouts(args.self_test,args.seed_start,args.test_output)
            if any(x['success_count']<args.self_test for x in report['summaries']):
                raise SystemExit(1)
        else:
            report=self_test(args.self_test,args.seed_start,args.test_output,layout)
            if report['summary']['success_count']<args.self_test:raise SystemExit(1)
        return
    if not args.robot_id: parser.error('请提供 --robot-id 参赛队号')
    out=Path(args.output) if args.output else DEFAULT_RUN_OUTPUT/(
        layout['layout_name']+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print(f"问题3多布局滚动路径版：布局={layout['layout_name']}，外围点数={layout['ring_points']}，"
          f"半径={layout['ring_radius']:g}米。请确保官方模拟器接口已就绪。日志目录：{out.resolve()}")
    result=Robot(HttpBackend(args.base_url),args.robot_id,out,layout=layout).run()
    print("===== 最终结果 =====")
    print(f"实际清除总数: {result['cleared_count']}")
    print(f"实际消耗总时间: {result['virtual_time_s']:.2f} s")
    if result['average_clear_time_s'] is None:
        print("实际平均消耗时间: N/A")
    else:
        print(f"实际平均消耗时间: {result['average_clear_time_s']:.2f} s")

    if result['status']!='complete':
        print('本次未确认全部完成；请保留日志并检查模拟器界面。')
        raise SystemExit(1)
if __name__=='__main__':
    main()
