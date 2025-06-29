from scsplanning import biconvex, check_problem_data
from pydrake.all import (ConvexSet,
                         HPolyhedron,
                         BezierCurve, 
                         CompositeTrajectory)
from pybezier import CompositeBezierCurve
import numpy as np
from typing import List, Tuple, Union
import time
import pycsdecomp as csd

class CollisionChecker:
    def __init__(self, 
                 mplant : csd.MinimalPlant, 
                 robot_geometries : List[int], 
                 ):
        self.mplant = mplant
        self.robot_geometries = robot_geometries

    def CheckConfigsCollisionFree(self, 
                                  configs: np.ndarray, 
                                  voxels : csd.Voxels, 
                                  voxel_radius : float):
        return csd.CheckCollisionFreeVoxelsCuda(configs, 
                                                voxels, 
                                                voxel_radius,
                                                self.mplant,
                                                self.robot_geometries)

class RegionCorrector:
    def __init__(self, 
                 mplant : csd.MinimalPlant, 
                 robot_geometries : List[int],
                 step_back : float = 0.01
                 ):
            self.mplant = mplant
            self.robot_geometries = robot_geometries
            self.options = csd.EditRegionsOptions()
            self.options.configuration_margin = step_back
    def correct(self, 
                collisions : np.ndarray,
                line_segment_idxs : List[int],
                edge_start_points : np.ndarray,
                edge_end_points : np.ndarray,
                regions : List[HPolyhedron],
                voxels : csd.Voxels,
                voxel_radius : float
                ):
        result = csd.EditRegionsCuda(collisions,
                                     line_segment_idxs,
                                     edge_start_points,
                                     edge_end_points,
                                     regions,
                                     self.mplant,
                                     self.robot_geometries,
                                     voxels,
                                     voxel_radius,
                                     self.options)
        #convert to drake regions
        regions_drake = [HPolyhedron(r.A(), r.b()) for r in result[0]]
        return regions_drake
    
def MintimeSCSWithPathFixing(start : np.ndarray,
                            goal : np.ndarray,
                            regions : List[HPolyhedron],
                            edges : List[np.ndarray],
                            ###########
                            vel_limits : List[np.ndarray],
                            acc_limits : List[np.ndarray],
                            ###########
                            checker : CollisionChecker,
                            corrector : RegionCorrector,
                            edge_inflator : csd.CudaEdgeInflator,
                            voxels : csd.Voxels,
                            voxel_radius : float,
                            max_attempts : int = 20,
                            degree : int = 6
                            ):
    attempt = 0
    time_scs = 0
    time_region_fixing = 0
    edge_start_points = []
    edge_end_points = []
    for e in edges:
        edge_start_points.append(e[0])
        edge_end_points.append(e[0])
    edge_start_points = np.array(edge_start_points)
    edge_end_points = np.array(edge_end_points)
    vel_set = HPolyhedron.MakeBox(vel_limits[0], vel_limits[1])
    acc_set = HPolyhedron.MakeBox(acc_limits[0], acc_limits[1])

    while attempt< max_attempts:
        tic = time.time()
        traj, cost = solve_scs_trajopt(q_init = start,
                                       q_term= goal,
                                       regions = [r for r in regions if r is not None],
                                       vel_set = vel_set,
                                       acc_set = acc_set,
                                       deg = degree,
                                       tol = 1e-2
                                       )
        if cost == np.inf:
            timing_info = {'trajopt': np.inf, 
                           'regionfixing': np.inf}
            return traj, cost, timing_info, False, False, []
        toc = time.time()
        time_scs+=toc-tic
        times = np.linspace(traj.start_time(), traj.end_time(), 8000)
        wps = traj.vector_values(times)
        wp_col_free = np.array(checker.CheckConfigsCollisionFree(wps, 
                                                                 voxels,
                                                                 voxel_radius))
        traj_col_free = np.all(wp_col_free)
        collisions = []
        if traj_col_free:
            break
        col_idx =np.where(1-1.0*wp_col_free)[0]
        collisions = wps[:, col_idx]
        time_cols = times[col_idx]
        line_seg_idxs = [traj.get_segment_index(t) for t in time_cols]

        regions = corrector.correct(collisions,
                                    line_seg_idxs,
                                    edge_start_points.T,
                                    edge_end_points.T,
                                    regions,
                                    voxels,
                                    voxel_radius
                                    )
        
        #ensure that PWL paths still covered by regions after edits otherwise add new regions
        for i, e in enumerate(edges):
            in_regions, reg = edge_in_regions(e, regions)
            if not in_regions:
                ccr = edge_inflator.inflateEdge(e[0], 
                                                e[1], 
                                                voxels, 
                                                voxel_radius,
                                                verbose = False)
                            
                region = check_region_and_correct(HPolyhedron(ccr.A(), ccr.b()), e)
                regions[i] = region

        print(f"[MintimeSCS] COllision found, attempted correction {attempt}.")
        attempt += 1
        toc2 = time.time()
        time_region_fixing += toc2 - toc
    if attempt == 0:
        first_solve_collision_free = True
    else:
        first_solve_collision_free = False
    timing_info = {'trajopt': time_scs, 
                   'regionfixing': time_region_fixing}
    sol_info = {'cost': cost,
                'timin_info': timing_info,
                'traj_col_free': traj_col_free,
                'first_solve_col_free': first_solve_collision_free,
                'collisions' : collisions}
    return traj, sol_info

def CSD_inflate_edges_given_pwl_path(path: List[np.ndarray],
                                     edge_inflator : csd.CudaEdgeInflator,
                                     csd_voxels : csd.Voxels,
                                     vox_radius : float,
                                     existing_regions : Union[None, List[HPolyhedron]] = None,
                                     verbose: bool = True,
                                     set_containment_tol = 1e-6
                                     ):
    regions = []
    edges = []
    path = np.array(path)
    if existing_regions is None:
        existing_regions = []

    for i in range(len(path)-1):
        edges.append(np.array([path[i], path[i+1]]))

    for e in edges:
        in_regions, reg = edge_in_regions(e, regions + existing_regions, tol=set_containment_tol/10)
        if in_regions:
            regions.append(None)
        else:
            ccr = edge_inflator.inflateEdge(e[0], 
                                            e[1], 
                                            csd_voxels, 
                                            vox_radius,
                                            verbose = verbose)
            region = check_region_and_correct(HPolyhedron(ccr.A(), 
                                                          ccr.b()),
                                              e,
                                              tol = set_containment_tol)
            assert region.PointInSet(e[0]) and region.PointInSet(e[1])
            regions.append(region)
    return regions, edges

def composite_bezier_to_drake(composite_curve : CompositeBezierCurve):
    segments = []
    for c in composite_curve.curves:
        segments.append(BezierCurve(c.initial_time,
                                    c.final_time, 
                                    (c.points).astype(np.float32).T))

    return CompositeTrajectory(segments)

def solve_scs_trajopt( q_init: np.ndarray,
                       q_term: np.ndarray,
                       regions: List[ConvexSet],
                       vel_set: ConvexSet,
                       acc_set: ConvexSet,
                       deg: int,
                       tol: float = 1e-3
                       ) -> Tuple[CompositeTrajectory, float]:
    
    try:
        traj = composite_bezier_to_drake(biconvex(q_init, 
                                                q_term, 
                                                regions, 
                                                vel_set, 
                                                acc_set, 
                                                deg,
                                                tol))

        cost = traj.end_time() - traj.start_time()
    except:
        print('[SCSTRAJOPT FAILED]')
        check_problem_data(q_init,
                           q_term,
                           regions,
                           vel_set, 
                           acc_set,
                           deg,
                           interior_tol=1e-5)
        traj = None
        cost = np.inf
    return traj, cost

def check_region_and_correct(region: HPolyhedron,
                             edge: List[np.ndarray],
                             tol = 0):
    
    vals = region.A()@edge[0] - region.b()
    ind = 1.*(region.A()@edge[0] - region.b()>=tol)
    correction1 = vals*ind
    vals2 = region.A()@edge[1] - region.b()
    ind2 = 1.*(region.A()@edge[1] - region.b()>=tol)
    correction2 = vals2*ind2
    region = HPolyhedron(region.A(), region.b() + correction1+correction2)
    if np.any(correction1) or np.any(correction2):
        print(f'CSPACE MARGIN ERROR DETECTED {correction1} {correction2}')
    return region
    
def edge_in_regions(edge, regions: List[HPolyhedron], tol = 1e-8):
    for r in regions:
        if r is not None:
            if r.PointInSet(edge[0,:], tol=tol) and r.PointInSet(edge[1,:], tol=tol):
                return True, r
    return False, []