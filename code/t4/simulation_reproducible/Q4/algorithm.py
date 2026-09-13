#!/usr/bin/env python3
"""问题四B版：位置—类型—朝向—固定接收半径联合估计。
Python 3.9+，依赖 numpy scipy。
运行：python rolling21_q4.py --robot-id 参赛队号
本地：python rolling21_q4.py --self-test 10 --test-output q4_tests
压力：python rolling21_q4.py --self-test 3 --source-mode outward --edge
固定布局：中心＋995米8点内圈＋1865米12点外圈，运行前验证全方向覆盖。
无信号更新联合候选状态及其条件半径区间；不直接删除硬位置圆盘，探索次数受限。
成功测站的凸组合保证定向/全向接收；其他候选明确为探索性补测。
方向概率和高斯信息量均为调度代理，不构成几何证书。
光学清除独立于发射方向；保留25米网格兜底。时限内未完成会如实报告。
本地环境为合成验证，不是官方模拟器，不读取官方隐藏数据。
"""
import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial import ConvexHull, QhullError
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

DELTA = math.radians(1.006)  # ±1度加两位小数取整裕量
N_CIRCLE = 96

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
    v=np.asarray(poly,dtype=float)
    center=v.mean(axis=0)
    tri=np.array([[center,a,b] for a,b in zip(v,np.roll(v,-1,axis=0))])
    for _ in range(depth):
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
        tri=np.concatenate([np.stack(x,axis=1) for x in
            [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]])
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
    triangles,areas=mesh(poly,quad_depth);points=triangles.mean(axis=1)
    if not len(points) or not np.all(np.isfinite(points)):return math.inf
    vertices=np.asarray(poly,dtype=float)
    depths=np.full(len(points),np.inf)
    for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
        depths=np.minimum(depths,point_segment_distances(points,a,b))
    d_star=float(np.max(depths))
    if not math.isfinite(d_star) or d_star<=1e-12:return math.inf
    cfg=Q12_LINK_CONFIG
    weights=areas*(cfg['eps_w']+(1-cfg['eps_w'])*(np.maximum(depths,0.)/d_star)**cfg['p_w'])
    u=points-np.asarray(station1,dtype=float);v=points-np.asarray(candidate,dtype=float)
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

def gaussian_moments(points,weights):
    w=np.asarray(weights,dtype=float);total=float(w.sum())
    if not math.isfinite(total) or total<=1e-300:raise ValueError('后验积分权重退化')
    w=w/total;mu=w@points;diff=points-mu
    covariance=(diff*w[:,None]).T@diff
    values,vectors=np.linalg.eigh((covariance+covariance.T)/2)
    covariance=(vectors*np.maximum(values,1e-4))@vectors.T
    return mu,covariance

def radius_interval(points,positives,negatives):
    lower=np.full(len(points),1000.);upper=np.full(len(points),1500.)
    for q in positives:lower=np.maximum(lower,np.linalg.norm(points-q,axis=1))
    for q in negatives:upper=np.minimum(upper,np.linalg.norm(points-q,axis=1))
    return lower,upper

def update_gaussian(target,q,angle):
    """受几何支持集约束的假定密度滤波(ADF)。
    p_new(G) ∝ N(G;mu,Sigma)*测角似然*接收半径条件似然*I(G在可行域)。
    对非高斯后验积分求均值/协方差，再用二维高斯近似。
    首次更新采用圆域内均匀先验；R~U[1000,1500]且固定，跨观测共享。
    第四问仅保留阳性接收的半径下界；阴性与朝向似然不进入位置矩更新。
    """
    triangles,areas=mesh(target['poly'],BAYES_CONFIG['quad_depth'])
    points=triangles.mean(axis=1)
    positives=target['stations'];negatives=[] # 定向负观测不能转成接收半径上界
    lower,upper=radius_interval(points,positives,negatives)
    new_width=np.maximum(upper-lower,0.)
    delta=points-np.asarray(q);distance=np.linalg.norm(delta,axis=1)
    residual=(np.arctan2(delta[:,1],delta[:,0])-math.radians(angle)+math.pi)%(2*math.pi)-math.pi
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    logw=np.log(areas)-residual**2/(2*variance)
    if 'mean' in target:
        diff=points-target['mean'];inv=np.linalg.inv(target['covariance'])
        logw-=0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
        old_lower,old_upper=radius_interval(points,positives[:-1],negatives)
        old_width=np.maximum(old_upper-old_lower,0.)
        likelihood=np.divide(new_width,old_width,out=np.zeros_like(new_width),where=old_width>1e-10)
    else:
        # 仅结合有效示向的接收半径下界，阴性不作为位置似然。
        likelihood=new_width/500.
    valid=(distance>5)&(likelihood>0)&(np.linalg.norm(points,axis=1)<=1800+1e-6)
    for station in positives:valid &= np.linalg.norm(points-np.asarray(station),axis=1)>5
    failure_support=np.ones(len(points),dtype=bool)
    for failed in target.get('failed_clears',[]):failure_support &= np.linalg.norm(points-np.asarray(failed),axis=1)>20
    valid &= failure_support
    note='gaussian_moment_projection'
    if np.any(valid):
        logw[valid]+=np.log(likelihood[valid]);logw[~valid]=-np.inf
        weights=np.exp(logw-np.max(logw))
    else:
        # 极窄/边界区域可能被有限求积漏采；回到几何区域的保守矩近似，绝不判空。
        weights=areas*failure_support
        if weights.sum()<=0:weights=areas
        note='geometric_moment_fallback'
    mu,cov=gaussian_moments(points,weights)
    target['mean']=mu;target['covariance']=cov
    return {'mean':mu.tolist(),'covariance':cov.tolist(),'note':note,
            'quadrature_points':len(points),'noise_std_deg':BAYES_CONFIG['noise_deg'],
            'gaussian_entropy_nats':float(math.log(2*math.pi*math.e)+0.5*np.linalg.slogdet(cov)[1])}

def update_unknown_belief(belief,coverage_map,ch):
    # 不把未收信号解释为目标在1000米圆盘外；存在性仅由完整21点扫描认证。
    return {'mean':belief['mean'].tolist(),'covariance':belief['covariance'].tolist(),
            'existence_probability':belief['existence'],
            'note':'directional_negative_recorded_without_position_exclusion'}

def binary_entropy(p):
    if p<=0 or p>=1:return 0.
    return -p*math.log(p)-(1-p)*math.log1p(-p)

def reception_hull_certified(stations,p):
    """成功测站凸包内保持固定半径圆盘及发射半平面内，二者均凸。"""
    a=np.unique(np.asarray(stations,dtype=float).reshape(-1,2),axis=0)
    if not len(a):return False
    p=np.asarray(p,dtype=float)
    if len(a)>=3:
        try:
            eq=ConvexHull(a).equations
            return bool(np.max(eq[:,:2]@p+eq[:,2])<=1e-8)
        except QhullError:pass
    i,j=max(((i,j) for i in range(len(a)) for j in range(len(a))),key=lambda ij:np.linalg.norm(a[ij[0]]-a[ij[1]]))
    v=a[j]-a[i];den=float(v@v)
    if den<1e-16:return bool(np.linalg.norm(p-a[i])<1e-8)
    t=float((p-a[i])@v/den)
    return bool(-1e-12<=t<=1+1e-12 and np.linalg.norm(p-a[i]-t*v)<1e-8)

def direction_proxy(target,p):
    # 在位置均值处对固定朝向做角度求积；仅是启发式成功概率。
    mu=np.asarray(target['mean']);angles=np.arange(360)*math.pi/180
    normals=np.c_[np.cos(angles),np.sin(angles)]
    positive=np.asarray(target['stations'])-mu
    allowed=np.all(normals@positive.T>=-1e-8,axis=1)
    omni_possible=True
    for q in target.get('negatives',[]):
        v=np.asarray(q)-mu
        if np.linalg.norm(v)<1000:
            allowed &= normals@v<0
            omni_possible=False
    if not allowed.any():return .5
    frac=float(np.mean(normals[allowed]@(np.asarray(p)-mu)>=0))
    # 0.5全向先验为选址假设，不据此判定真实源类型。
    return max(.05,min(.99,(.5+.5*frac) if omni_possible else frac))

def bayes_select(poly,stations,current,target,deadline):
    """先保证原有距离强约束；辐射接收可认证则认证，否则明确探索并限次。"""
    begin=time.monotonic();mu=np.asarray(target['mean']);cov=target['covariance']
    st=np.asarray(stations);tri,_=mesh(poly,2)
    station_r2=np.sum((tri[:,:,None,:]-st[None,None,:,:])**2,axis=3)
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    candidates=[]
    for origin in (mu,np.asarray(target['center'])):
        for radius in (25.,50.,100.,200.,400.):
            for theta in np.arange(16)*math.pi/8:
                candidates.append(origin+radius*np.array([math.cos(theta),math.sin(theta)]))
    for q in st:
        for fraction in (.25,.5,.75,.9):candidates.append((1-fraction)*q+fraction*mu)
    for i in range(len(st)):
        for j in range(i):
            for t in (.25,.5,.75):candidates.append((1-t)*st[i]+t*st[j])
    rows=[];seen=set()
    for p in candidates:
        if time.monotonic()>deadline-20:raise BudgetStop('第四问选址接近截止时间')
        key=tuple(np.round(p,6))
        if key in seen:continue
        seen.add(key)
        if np.min(np.linalg.norm(st-p,axis=1))<5 or np.linalg.norm(p-mu)<20:continue
        if any(dist(p,q)<5 for q in target.get('negatives',[])):continue
        d2=np.sum((tri-p)**2,axis=2)
        upper=float(np.max(np.minimum(np.max(d2-1e6,axis=1),np.min(np.max(d2[:,:,None]-station_r2,axis=1),axis=1))))
        certified=reception_hull_certified(st,p)
        if not certified and upper> -1e-4:continue
        delta=mu-p;r2=float(delta@delta);h=np.array([-delta[1],delta[0]])/r2
        info=.5*math.log1p(max(0.,float(h@cov@h))/variance)
        probability=1. if certified else direction_proxy(target,p)
        duration=10+dist(current,p)/5+.5*dist(p,mu)/5
        score=probability*info/duration
        rows.append(dict(p=tuple(float(x) for x in p),score=score,info=info,probability=probability,
                         duration=duration,certified=certified,range_upper=upper))
    if not rows:return None,{'reason':'no_feasible_new_point','reception_certificate':{'certified':False}}
    original=max(rows,key=lambda x:x['score']);chosen=original;cfg=Q12_LINK_CONFIG
    tol=min(cfg['tau'],cfg['max_primary_loss_ratio'],cfg['primary_tie_rel_tol'])
    tied=[r for r in rows if (original['score']-r['score'])/max(original['score'],1e-12)<=tol]
    olddop=depth_weighted_dop(poly,stations[0],original['p'])
    bestdop=olddop
    for row in tied:
        dop=depth_weighted_dop(poly,stations[0],row['p'])
        if cfg['enable_dop_override'] and math.isfinite(olddop) and dop<bestdop and dop<=olddop*(1-cfg['min_dop_improvement']):
            chosen=row;bestdop=dop
    cert={'certified':chosen['certified'],
          'method':'convex_combination_of_positive_stations' if chosen['certified'] else 'exploratory_direction_unknown',
          'range_certified':True}
    return chosen['p'],{'reason':'selected','position':chosen['p'],
        'expected_information_nats':chosen['probability']*chosen['info'],
        'conditional_bearing_information_nats':chosen['info'],
        'detection_probability_proxy':chosen['probability'],
        'information_per_second':chosen['score'],'estimated_action_chain_s':chosen['duration'],
        'reception_certificate':cert,'elapsed_s':time.monotonic()-begin,
        'problem2_dop_tiebreak':{'override':chosen is not original,'candidate_count':len(rows),
             'original_dop':olddop if math.isfinite(olddop) else None,
             'chosen_dop':bestdop if math.isfinite(bestdop) else None},
        'note':'position_mean_orientation_proxy; uncertified_direction_candidates_may_return_no_signal'}

ROUTE_MEASURE_CONFIG={'min_information_ratio':0.70,'min_saving_s':10.0,
                      'segment_fractions':(0.25,0.5,0.75,1.0),
                      'min_stop_information_nats':0.50,'max_target_channels_per_stop':6}

def mandatory_stop_measurement_score(target,q):
    """评价必经扫描点是否适合作为已发现目标的顺路补测点。"""
    q=np.asarray(q,dtype=float);poly=target.get('poly') or []
    if not poly or 'mean' not in target or 'covariance' not in target:return None
    # 对凸定位区域，距固定点的最大值在顶点取得；距离约束之外还需成功测站凸包的辐射方向认证。
    receive_bound=max(np.linalg.norm(np.asarray(v,dtype=float)-q) for v in poly)
    if receive_bound>1000-1e-6 or not reception_hull_certified(target['stations'],q):return None
    stations=np.asarray(target.get('stations',[]),dtype=float)
    if len(stations) and float(np.min(np.linalg.norm(stations-q,axis=1)))<5-1e-7:return None
    mu=np.asarray(target['mean'],dtype=float);delta=mu-q;r2=float(delta@delta)
    if r2<20**2:return None
    h=np.array([-delta[1],delta[0]])/r2
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    info=0.5*math.log1p(max(0.,float(h@np.asarray(target['covariance'])@h))/variance)
    if info<ROUTE_MEASURE_CONFIG['min_stop_information_nats']:return None
    return {'information_nats':info,'receive_bound_m':float(receive_bound)}

def route_aligned_measurement(poly,stations,current,target,next_stop,free_point,free_detail):
    """在当前点到下一必经点的线段上寻找可替代自由补测点。

    仅当候选点满足保守接收条件、与既有测站/目标估计保持间距、信息量达到
    自由最优点的指定比例，并且预计“补测后去中心再回到下一节点”的总链路
    至少节省指定秒数时采用。这样不会仅为贴路线而牺牲过多交会质量。
    """
    if next_stop is None or free_point is None:return free_point,{**free_detail,'route_aligned':False,'route_reason':'no_next_stop'}
    mu=np.asarray(target['mean'],dtype=float);cov=np.asarray(target['covariance'],dtype=float)
    stations=np.asarray(stations,dtype=float);current=np.asarray(current,dtype=float)
    next_stop=np.asarray(next_stop,dtype=float);free_point=np.asarray(free_point,dtype=float)
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    triangles,_=mesh(poly,2)
    station_r2=np.sum((triangles[:,:,None,:]-stations[None,None,:,:])**2,axis=3)
    def receive(p):
        d2=np.sum((triangles-p)**2,axis=2)
        upper=np.minimum(np.max(d2-1e6,axis=1),np.min(np.max(d2[:,:,None]-station_r2,axis=1),axis=1))
        return float(upper.max())
    def information(p):
        delta=mu-p;r2=float(delta@delta)
        if r2<1e-8:return 0.
        h=np.array([-delta[1],delta[0]])/r2
        return 0.5*math.log1p(max(0.,float(h@cov@h))/variance)
    def feasible(p):
        return (reception_hull_certified(stations,p) and receive(p)<=-1e-4 and np.linalg.norm(p-mu)>=20-1e-7 and
                float(np.min(np.linalg.norm(p-stations,axis=1)))>=5-1e-7)
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
                      'reception_certificate':{'certified':True,'method':'convex_combination_of_positive_stations'},
                      'note':'feasible_point_on_current_to_next_planned_stop'}
    return tuple(free_point),{**free_detail,'route_aligned':False,
                              'route_reason':'insufficient_information_or_saving',
                              'best_route_information_ratio':ratio,
                              'best_route_estimated_saving_s':saving,
                              'next_stop':tuple(float(v) for v in next_stop)}

PROBE_CONFIG={'max_attempts':5,'min_probability':0.2,'max_radius':80.}

JOINT_CONFIG={'max_detour_m':5.,'horizon':3,'min_saving_s':0.,'candidate_limit':12}

def probe_distribution(target):
    """失败clear后保留条件分布的非高斯形状，不反复用高斯投影重加权。
    新测向到来时(测点数改变)重新以新高斯工作后验生成求积权重。
    """
    version=len(target.get('stations',[]))
    cached=target.get('probe_base')
    if cached is None or cached['version']!=version:
        triangles,areas=mesh(target['poly'],4);points=triangles.mean(axis=1)
        diff=points-target['mean'];inv=np.linalg.inv(target['covariance'])
        logw=np.log(areas)-0.5*np.einsum('ij,jk,ik->i',diff,inv,diff)
        weights=np.exp(logw-np.max(logw));weights/=weights.sum()
        cached={'version':version,'points':points,'weights':weights};target['probe_base']=cached
    points=cached['points'];weights=cached['weights'].copy()
    for q in target.get('failed_clears',[]):weights[np.linalg.norm(points-q,axis=1)<=20]=0.
    if weights.sum()<=1e-300:return None
    weights/=weights.sum()
    return points,weights

def joint_probe_plan(target,current,current_channel,ch,anchor,same_target,next_stop):
    """在实际即将执行的路段上插入1~3次clear，枚举成功/全部失败分支。
    返回值为启发式预计节省，不是无条件时间上界；只执行第一步，反馈后滚动重算。
    """
    remaining_budget=PROBE_CONFIG['max_attempts']-target.get('probe_attempts',0)
    if remaining_budget<=0 or target['radius']>PROBE_CONFIG['max_radius'] or target.get('planned') is None:return None
    sample=probe_distribution(target)
    if sample is None:return None
    points,weights=sample;current=np.asarray(current,dtype=float);anchor=np.asarray(anchor,dtype=float)
    vector=anchor-current;length2=float(vector@vector);direct=dist(current,anchor)
    def project(q):
        if length2<=1e-12:return current.copy()
        return current+np.clip(float((np.asarray(q)-current)@vector)/length2,0,1)*vector
    seeds=[tuple(current),tuple(project(target['mean'])),tuple(project(target['center']))]
    if length2>1e-12:
        direction=vector/math.sqrt(length2);center=project(target['mean'])
        for offset in (-40,-20,-10,10,20,40):seeds.append(tuple(project(center+offset*direction)))
    poly=np.asarray(target['poly']);low=poly.min(axis=0);high=poly.max(axis=0)
    n=np.maximum(1,np.ceil((high-low)/25).astype(int))
    seeds.extend([tuple(target['mean']),tuple(target['center'])])
    for iy in range(n[1]):
        for ix in range(n[0]):
            q=low+(np.array([ix,iy])+.5)*(high-low)/n
            seeds.extend([tuple(q),tuple(project(q))])
    candidates=[];seen=set()
    for q in seeds:
        key=tuple(round(x,5) for x in q)
        if key in seen:continue
        seen.add(key)
        if any(dist(q,p)<1 for p in target.get('failed_clears',[])):continue
        extra=dist(current,q)+dist(q,anchor)-direct
        if extra>JOINT_CONFIG['max_detour_m']+1e-7:continue
        mask=np.linalg.norm(points-q,axis=1)<=20
        mass=float(weights[mask].sum())
        if mass<0.01:continue
        candidates.append({'q':tuple(q),'mask':mask,'mass':mass,'extra':max(0.,extra)})
    candidates.sort(key=lambda c:c['mass']/(3+dist(current,c['q'])/5),reverse=True)
    candidates=candidates[:JOINT_CONFIG['candidate_limit']]
    if not candidates:return None
    planned=target['planned'];center=target['center']
    if same_target:
        # baseline: current->测向点->估计清除点->next_stop。
        tail=5+(current_channel!=ch)+dist(anchor,center)/5+5
        if next_stop is not None:tail+=dist(center,next_stop)/5
        success_tail=lambda q:dist(q,next_stop)/5 if next_stop is not None else 0.
    else:
        # 顺路处理其他目标：保留既定路线，估算避免将来专门定位该目标的成本。
        tail=dist(anchor,planned)/5+5+(current_channel!=ch)+dist(planned,center)/5+5
        success_tail=lambda q:dist(q,anchor)/5
    baseline=direct/5+tail
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
            total=expected+returned+after*(dist(q,anchor)/5+tail)
            sequence=order+[i]
            saving=baseline-total
            if saving>=JOINT_CONFIG['min_saving_s'] and (best is None or saving>best['expected_saving_s']):
                best={'position':candidates[sequence[0]]['q'],
                      'sequence':[candidates[j]['q'] for j in sequence],
                      'probability':.995*candidates[sequence[0]]['mass'],
                      'sequence_success_probability':1-after,'expected_time_s':total,
                      'reference_time_s':baseline,'expected_saving_s':saving,
                      'worst_branch_extra_distance_m':max(0.,detour),'anchor':tuple(anchor),
                      'same_target':same_target,'hypothesis':'approximate_posterior_and_one_measurement_continuation'}
            visit(sequence,excluded|candidate['mask'],after,route_length,expected,q,returned)
    visit([],np.zeros(len(points),dtype=bool),1.,0.,0.,tuple(current),0.)
    return best

def condition_failed_clear(target,q):
    target.setdefault('failed_clears',[]).append(tuple(q))
    sample=probe_distribution(target)
    if sample is None:return {'note':'quadrature_missed_remaining_region_keep_geometric_outer_bound'}
    target['mean'],target['covariance']=gaussian_moments(*sample)
    target['planned']=None
    return {'note':'posterior_conditioned_on_distance_greater_than_20',
            'mean':target['mean'].tolist(),'covariance':target['covariance'].tolist()}

def certify(p,radius=1.,minside=1e-5):
 stack=[(0.,0.,1.8)]; count=0; maxdepth=0
 while stack:
  x,y,h=stack.pop();count+=1
  if np.hypot(max(abs(x)-h,0),max(abs(y)-h,0))>1.8:continue
  corn=np.array([[x-h,y-h],[x-h,y+h],[x+h,y-h],[x+h,y+h]])
  ok=np.max(np.sum((p[:,None,:]-corn[None,:,:])**2,axis=2),axis=1)<(radius-1e-10)**2
  q=p[ok]
  if len(q)>=3:
   try:
    eq=ConvexHull(q).equations
    if np.max(corn@eq[:,:2].T+eq[:,2])<-1e-10:continue
   except:pass
  if h*2<minside:return dict(pass_all=False,box=[x,y,h],boxes=count)
  hh=h/2
  stack.extend((x+dx*hh,y+dy*hh,hh) for dx in [-1,1] for dy in [-1,1])
 return dict(pass_all=True,boxes=count)

LAYOUT_PRESETS={'q4_21':{'ring_points':20,'ring_radius':1865.}}
DEFAULT_LAYOUT='q4_21'

def scan_points(ring_points=20,ring_radius=1865.):
    if ring_points!=20 or ring_radius!=1865.:raise ValueError('第四问仅接受已验证的21点布局')
    return [(0.,0.)]+[(r*math.cos(2*math.pi*k/n),r*math.sin(2*math.pi*k/n))
                       for n,r in [(8,995.),(12,1865.)] for k in range(n)]

def resolve_layout(layout_name=DEFAULT_LAYOUT,ring_points=None,ring_radius=None):
    if layout_name!=DEFAULT_LAYOUT or ring_points is not None or ring_radius is not None:
        raise ValueError('第四问固定21点，不接受自定义单环参数')
    check=certify(np.asarray(scan_points())/1000.,radius=.999)
    if not check['pass_all']:raise ValueError('21点连续覆盖验证未通过')
    return {'layout_name':DEFAULT_LAYOUT,'ring_points':20,'ring_radius':1865.,
            'inner_points':8,'inner_radius':995.,'outer_points':12,'outer_radius':1865.,
            'certificate':{'certified':True,'method':'adaptive_box_local_convex_hull_float_certificate',
                           'target_radius_m':1800.,'verified_reception_radius_m':999.,
                           'all_emission_directions':True,'boxes':check['boxes'],
                           'note':'double precision with conservative tolerance; not interval arithmetic'}}

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
    与目标圆相交的方格全部保留；定向no_signal不删除任何格。
    格心仅为位置矩辅助，不作为真实目标离散位置或缺席证书。
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
        # 不删除任何位置格；定向角未知，单次阴性不是位置排除证据。
        self.observations[ch-1]+=1
        return 0

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

LOCAL_CONFIG={'source_mode':'mixed','edge':False}

class LocalBackend:
    """独立生成10~16个全向/定向混合源；同频道同位置误差固定，无真值传给策略。"""
    def __init__(self,seed,edge=False):
        edge=edge or LOCAL_CONFIG['edge']
        rng=random.Random(seed)
        self.seed=seed
        self.targets={}
        for ch in rng.sample(range(1,21),rng.randint(10,16)):
            r=1800 if edge else 1800*math.sqrt(rng.random())
            t=rng.random()*2*math.pi
            self.targets[ch]=((r*math.cos(t),r*math.sin(t)),1000 if edge else rng.uniform(1000,1500))
        orientation_rng=random.Random(seed+910921)
        mode=LOCAL_CONFIG['source_mode']
        keys=sorted(self.targets); self.directions={}
        for i,ch in enumerate(keys):
            directional=(mode in ('directional','outward') or (mode=='mixed' and i%2==0))
            p=self.targets[ch][0]
            self.directions[ch]=(math.atan2(p[1],p[0]) if mode=='outward' else orientation_rng.uniform(0,2*math.pi)) if directional else None
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
                theta=self.directions.get(ch)
                visible=(theta is None or (q[0]-target[0][0])*math.cos(theta)+(q[1]-target[0][1])*math.sin(theta)>=-1e-9)
                kind='no_signal' if not target or d>target[1] or not visible else ('near' if d<=5 else 'direction')
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
            raise RuntimeError('请求被拒绝 accepted=false；确认参赛队号、接口就绪状态。')
        self.vt=float(result['virtual_time_s'])
        if path=='/enter':
            self.deadline=time.monotonic()+float(result['remaining_real_duration_s'])
            self.vmax=float(result.get('max_virtual_duration_s',360000))
        if q is not None:
            self.distance+=dist(self.q,q); self.q=tuple(q)
        if path=='/measure':
            self.measures+=1; self.switches+=ch!=self.ch; self.ch=ch
            if result.get('measure_result') not in ('near','direction','no_signal'):
                raise RuntimeError('未知检测响应')
        if path=='/clear':
            self.clears+=1
            if result.get('clear_result') not in ('success','no_target_in_range'):
                raise RuntimeError('未知清除响应')
            if result['clear_result']=='success': self.done.add(ch)
        if not self.quiet and q is not None:
            kind=result.get('measure_result',result.get('clear_result'))
            print(f'{self.seq:4d} {path:8s} ch={ch:2d} ({q[0]:8.1f},{q[1]:8.1f}) {kind:20s} T={self.vt:.1f}s cleared={len(self.done)}',flush=True)
        return result
    def clear(self,ch,q):
        return self.call('/clear',q,ch)['clear_result']=='success'
    def clear_point(self, poly, center):
        """在20米共同覆盖圆盘交集中选离当前位置近的清除点。"""
        v=np.asarray(poly); current=np.asarray(self.q); scale=1000.
        def safe(z): return (19.8**2-np.sum((z*scale-v)**2,axis=1))/400.
        result=minimize(lambda z:float(np.sum((z-current/scale)**2)),np.asarray(center)/scale,
                        method='SLSQP',constraints=[{'type':'ineq','fun':safe}],
                        options={'maxiter':40,'ftol':1e-9})
        q=tuple(result.x*scale)
        return q if max(dist(q,p) for p in poly)<=19.80001 else center

    def make_fallback(self, target):
        """对当前保守凸区域作25米网格覆盖；失败后依次尝试剩余格。"""
        poly=target['poly']; cells=[]
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
                        belief=update_unknown_belief(self.beliefs[ch],self.coverage_map,ch)
                        self.write_log({'event':'negative_update','channel':ch,**belief})
                        # 仅尚未发现的频道可由全覆盖扫描认证为不存在；已发现目标的
                        # no_signal 只记录位置并限制重复探索，不得排除目标位置或误判为无源。
                        if ch in self.targets:
                            self.targets[ch].setdefault('negatives',[]).append(q)
                            self.targets[ch]['planned']=None
                            self.targets[ch]['negative_attempts']=self.targets[ch].get('negative_attempts',0)+1
                            joint_info=update_joint(self.targets[ch])
                            self.beliefs[ch].update(mean=self.targets[ch]['mean'],covariance=self.targets[ch]['covariance'])
                            self.write_log({'event':'joint_negative_update','channel':ch,**joint_info})
                        elif len(self.searched[ch])==len(scans):
                            self.absent.add(ch);self.beliefs[ch]['existence']=0.
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
                q=target['fallback'].pop(0);self.clear(ch,q)
                self.scheduler_counts['fallback_clears']+=1;continue
            if target.get('planned') is None:
                if len(target['stations'])+target.get('negative_attempts',0)>=8:self.make_fallback(target);continue
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
                else:
                    info=condition_failed_clear(target,probe['position'])
                    self.beliefs[ch].update(mean=target['mean'],covariance=target['covariance'])
                    self.write_log({'event':'failed_clear_update','channel':ch,**info})
                self.write_log({'event':'probe_result','channel':ch,'success':success,'plan':probe})
            else:
                res=self.call('/measure',q,ch);self.scheduler_counts['localization_measures']+=1
                if res['measure_result']=='no_signal':
                    target.setdefault('negatives',[]).append(tuple(q))
                    target['negative_attempts']=target.get('negative_attempts',0)+1
                    target['planned']=None
                    joint_info=update_joint(target)
                    self.beliefs[ch].update(mean=target['mean'],covariance=target['covariance'])
                    self.write_log({'event':'joint_negative_update','channel':ch,**joint_info})
                    self.write_log({'event':'directional_local_no_signal','channel':ch,'position':q,
                                    'note':'keep_geometric_region_and_replan'})
                else:self.accept_bearing(ch,q,res)
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
            self.call('/enter'); entered=True
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
                 'strategy':'q4_21_joint_position_orientation_v1',
                 'coverage_waypoints':21,'inner_ring':{'points':8,'radius_m':995},
                 'outer_ring':{'points':12,'radius_m':1865},
                 'directional_negative_count':sum(t.get('negative_attempts',0) for t in getattr(self,'targets',{}).values()),
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
                          'note':'方格仅为位置矩辅助；定向阴性不删除方格，缺席由完整21点扫描认证。'}
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
                        'reception_radius_m':value[1],'direction_rad':backend.directions[ch]}
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
    root=Path(output_dir or 'local_tests');root.mkdir(parents=True,exist_ok=True)
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
    root=Path(output_dir or 'layout_comparison');root.mkdir(parents=True,exist_ok=True)
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
    manifest={'experiment':'q4_fixed_layout_common_random_numbers',
              'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
              'description':'问题四混合源本地合成仿真；各布局使用完全相同的seed和目标真值。',
              'case_count':count,'seed_start':seed_start,'seeds':seeds,
              'presets':LAYOUT_PRESETS,
              'coverage_certificates':{x['layout_name']:x['certificate'] for x in resolved},
              'output_structure':'<layout_name>/case_NN/保存每局完整过程文件'}
    (root/'experiment_manifest.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summaries':summaries},ensure_ascii=False,indent=2))
    return results

JOINT_DIRECTION_CONFIG={'direction_bins':96,'position_depth':3,'predictive_states':1536,'prior_omni':.5}

def update_joint(target):
    """在当前硬几何多边形内重算全历史联合后验，固定半径区间解析积分。
    离散朝向/三角求积只影响调度，不删除硬几何支持集，不认证缺席。
    """
    cfg=JOINT_DIRECTION_CONFIG
    triangles,areas=mesh(target['poly'],cfg['position_depth']);points=triangles.mean(axis=1)
    n=len(points);angles=(np.arange(cfg['direction_bins'])+.5)*2*math.pi/cfg['direction_bins']
    normals=np.c_[np.cos(angles),np.sin(angles)];k=len(angles)
    lower=np.full(n,1000.);upper=np.full((n,k+1),1500.)
    valid=np.linalg.norm(points,axis=1)<=1800+1e-7
    orientation_valid=np.ones((n,k+1),dtype=bool)
    log_position=np.log(areas)
    variance=math.radians(BAYES_CONFIG['noise_deg'])**2
    for q,angle in target.get('bearing_history',[]):
        v=np.asarray(q)-points;distance=np.linalg.norm(v,axis=1)
        lower=np.maximum(lower,distance);valid &= distance>5
        predicted=np.arctan2(-v[:,1],-v[:,0])
        residual=(predicted-math.radians(angle)+math.pi)%(2*math.pi)-math.pi
        log_position-=residual**2/(2*variance)
        orientation_valid[:,1:] &= v@normals.T>=-1e-8
    for q in target.get('negatives',[]):
        v=np.asarray(q)-points;distance=np.linalg.norm(v,axis=1)
        emits=np.c_[np.ones(n,bool),v@normals.T>=-1e-8]
        upper=np.minimum(upper,np.where(emits,distance[:,None],1500.))
    for q in target.get('failed_clears',[]):valid &= np.linalg.norm(points-q,axis=1)>20
    width=np.maximum(upper-lower[:,None],0.)
    prior=np.r_[cfg['prior_omni'],np.full(k,(1-cfg['prior_omni'])/k)]
    position=np.exp(log_position-log_position.max())*valid
    weights=position[:,None]*width*orientation_valid*prior[None,:]/500.
    total=float(weights.sum())
    if not math.isfinite(total) or total<=1e-280:
        target.pop('joint_posterior',None)
        return {'note':'joint_quadrature_degenerate_keep_geometric_and_existing_moments'}
    weights/=total;marginal=weights.sum(axis=1)
    target['mean'],target['covariance']=gaussian_moments(points,marginal)
    # 确定性等质量压缩仅用于多候选点的接收概率预测。
    flat=weights.ravel();size=cfg['predictive_states']
    indexes=np.searchsorted(np.cumsum(flat),(np.arange(size)+.5)/size)
    ip,ia=np.unravel_index(np.minimum(indexes,len(flat)-1),weights.shape)
    target['joint_posterior']=dict(points=points,weights=marginal,
        predictive_points=points[ip],predictive_angles=ia,
        predictive_lower=lower[ip],predictive_upper=upper[ip,ia],normals=normals,
        omni_probability=float(weights[:,0].sum()))
    target.pop('probe_base',None)
    return {'note':'joint_position_orientation_fixed_radius_full_history_quadrature',
            'mean':target['mean'].tolist(),'covariance':target['covariance'].tolist(),
            'omni_probability':float(weights[:,0].sum()),'position_nodes':n,'direction_bins':k}

# 替换基版函数；硬几何更新仍由原 accept_bearing 执行。
_base_position_update=update_gaussian

def update_gaussian(target,q,angle):
    # 先保留基版作为求积退化时的工作矩回退，再以完整历史联合后验替换。
    result=_base_position_update(target,q,angle)
    target.setdefault('bearing_history',[]).append((tuple(q),float(angle)))
    result.update(update_joint(target))
    return result

def direction_proxy(target,p):
    posterior=target.get('joint_posterior')
    if posterior is None:return .5
    v=np.asarray(p)-posterior['predictive_points'];distance=np.linalg.norm(v,axis=1)
    ia=posterior['predictive_angles'];normal=posterior['normals'][np.maximum(ia-1,0)]
    emits=(ia==0)|(np.einsum('ij,ij->i',v,normal)>=-1e-8)
    lower=posterior['predictive_lower'];upper=posterior['predictive_upper']
    probability=np.maximum(upper-np.maximum(lower,distance),0)/np.maximum(upper-lower,1e-12)
    return float(np.clip(np.mean(probability*emits),.01,.999))

_base_probe_distribution=probe_distribution

def probe_distribution(target):
    posterior=target.get('joint_posterior')
    if posterior is None:return _base_probe_distribution(target)
    points=posterior['points'];weights=posterior['weights'].copy()
    for q in target.get('failed_clears',[]):weights[np.linalg.norm(points-q,axis=1)<=20]=0
    if weights.sum()<=1e-280:return None
    return points,weights/weights.sum()

_base_failed_clear=condition_failed_clear

def condition_failed_clear(target,q):
    result=_base_failed_clear(target,q)
    result.update(update_joint(target))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source-mode',choices=['mixed','omni','directional','outward'],default='mixed',help='仅本地：混合/全向/定向/朝外定向')
    parser.add_argument('--edge',action='store_true',help='仅本地：目标位于1800米边界，接收半径1000米')
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
                        help='预设布局，默认 q4_21')
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
    parser.add_argument('--test-output',default='local_tests',help='本地测试结果目录')
    parser.add_argument('--self-test',type=int,metavar='N',help='运行N个本地随机案例，不连接官方模拟器')
    args=parser.parse_args()
    LOCAL_CONFIG.update(source_mode=args.source_mode,edge=args.edge)
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
    out=Path(args.output) if args.output else Path('runs')/(
        layout['layout_name']+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    print(f"问题4定向混合滚动路径版：布局={layout['layout_name']}，外围点数={layout['ring_points']}，"
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
