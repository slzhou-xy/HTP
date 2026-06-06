from math import radians, cos, sin, asin, sqrt
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import scipy
from loguru import logger
from utils import pload
import shapely
from shapely.geometry import Point
from transformers import HfArgumentParser
from tqdm import tqdm


@dataclass
class ParserArguments:
    seed: int = field(default=42)
    exp_name: str = field(
        default='YOUR_NAME',
        metadata={"help": "Experiment name"}
    )
    city: str = field(default='porto')


def eval_road_density(road_trajs, road_edges):
    road_density = np.zeros((road_edges.shape[0],), dtype=int)

    for traj in road_trajs:
        new_road_traj = []
        for road in traj:
            if len(new_road_traj) == 0 or new_road_traj[-1] != road:
                new_road_traj.append(road)
        for road in new_road_traj:
            road_density[road] += 1
    return road_density


def eval_gps2road_distance(gps_trajs, road_trajs, road_edges):
    road_geoms = road_edges["geometry"].tolist()
    road_geoms = [shapely.from_wkt(geo) for geo in road_geoms]

    distances = []
    for gps_traj, road_traj in tqdm(zip(gps_trajs, road_trajs)):
        for (lon, lat), road in zip(gps_traj, road_traj):
            road_geometry = road_geoms[road]
            point = Point(lon, lat)
            proj_dist = road_geometry.project(point)
            nearest_point = road_geometry.interpolate(proj_dist)
            dist = geodistance(lon, lat, nearest_point.x, nearest_point.y)
            distances.append(dist)
    return distances


def in_bound(lon, lat, bounds):
    return bounds['min_lat'] < lat < bounds['max_lat'] and bounds['min_lon'] < lon < bounds['max_lon']


def geodistance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [float(lon1), float(lat1), float(lon2), float(lat2)])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371
    return distance


# get distance of a trajectory
def get_distance(traj):
    distance = 0
    segment_distance = []
    for i in range(len(traj) - 1):
        source = traj[i]
        target = traj[i + 1]
        value = geodistance(source[0], source[1], target[0], target[1])
        distance += value
        segment_distance.append(value)
    return distance, segment_distance


def get_geogradius(traj):
    rid_lon = [point[0] for point in traj]
    rid_lat = [point[1] for point in traj]
    if len(rid_lat) == 0:
        return 0
    lon1, lat1 = np.mean(rid_lon), np.mean(rid_lat)
    rad = []
    for i in range(len(rid_lat)):
        lon2 = rid_lon[i]
        lat2 = rid_lat[i]
        dis = geodistance(lon1, lat1, lon2, lat2)
        rad.append(dis)
    rad = np.mean(rad)
    return rad


def eval_radius(trajs):
    radii = []
    for traj in trajs:
        radii.append(get_geogradius(traj))
    return radii


def eval_grid_density(trajs, dataset, grid_size):
    if dataset == 'porto':
        bounds = {'min_lat': 41.13, 'max_lat': 41.19, 'min_lon': -8.69, 'max_lon': -8.55}
    else:
        bounds = {'min_lat': 30.56, 'max_lat': 30.79, 'min_lon': 103.92, 'max_lon': 104.21}

    lat_num = int(geodistance(bounds['min_lon'], bounds['min_lat'], bounds['min_lon'],
                              bounds['max_lat']) * 1000 // grid_size)
    lon_num = int(geodistance(bounds['min_lon'], bounds['min_lat'], bounds['max_lon'],
                              bounds['min_lat']) * 1000 // grid_size)

    grid_density = np.zeros(lat_num * lon_num)
    lat_dist = bounds['max_lat'] - bounds['min_lat']
    lon_dist = bounds['max_lon'] - bounds['min_lon']
    lat_size, lon_size = lat_dist / lat_num, lon_dist / lon_num

    for traj in trajs:
        unique_traj = []
        for point in traj:
            if not in_bound(point[0], point[1], bounds):
                continue
            grid_i = int((point[0] - bounds['min_lon']) / lon_size)
            grid_j = int((point[1] - bounds['min_lat']) / lat_size)
            grid = grid_j * lon_num + grid_i
            if len(unique_traj) == 0 or unique_traj[-1] != grid:
                unique_traj.append(grid)
        for grid in unique_traj:
            grid_density[grid] += 1
    return grid_density


def JS_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two probability distributions."""
    # Add small epsilon to avoid divide by zero issues
    epsilon = 1e-10
    p = np.asarray(p) + epsilon
    q = np.asarray(q) + epsilon

    # Normalize to ensure they're proper probability distributions
    p = p / np.sum(p)
    q = q / np.sum(q)

    M = (p + q) / 2
    return 0.5 * scipy.stats.entropy(p, M, base=2) + 0.5 * scipy.stats.entropy(q, M, base=2)


def calculate_jsd(real, generated, num_bins=None):
    """Calculate Jensen-Shannon divergence between two distributions with adaptive binning."""
    # Remove NaN values
    real = np.array(real)
    generated = np.array(generated)

    MIN = min(np.min(real), np.min(generated))
    MAX = max(np.max(real), np.max(generated))

    # Determine appropriate number of bins if not specified
    if num_bins is None:
        # Use Sturges' formula as a starting point
        num_bins = max(int(np.log2(len(real) + len(generated))) + 1, 5)
        num_bins = min(num_bins, 50)  # Cap at 50 bins

    # Ensure MIN and MAX are different
    if abs(MAX - MIN) < 1e-10:
        MAX = MIN + 1e-6

    bins = np.linspace(MIN - 1e-6, MAX + 1e-6, num=num_bins)

    # Calculate probability distributions
    PDF1 = pd.cut(real, bins).value_counts() / len(real)
    PDF2 = pd.cut(generated, bins).value_counts() / len(generated)

    # Ensure all bins are represented in both distributions
    all_indices = set(PDF1.index) | set(PDF2.index)
    for idx in all_indices:
        if idx not in PDF1:
            PDF1[idx] = 0
        if idx not in PDF2:
            PDF2[idx] = 0

    # Sort by index for consistent comparison
    PDF1 = PDF1.sort_index()
    PDF2 = PDF2.sort_index()

    return JS_divergence(PDF1.values, PDF2.values)


def eval_travel_and_segment_distance(trajs):
    travel_distances = []
    segment_distances = []
    for traj in trajs:
        distance, segment_distance = get_distance(traj)
        travel_distances.append(distance)
        segment_distances.extend(segment_distance)
    return travel_distances, segment_distances


def eval_pattern(real_density, generated_density, k):
    top_k_real = np.argsort(real_density)[::-1][:k]
    top_k_generated = np.argsort(generated_density)[::-1][:k]
    presicion = len(set(top_k_real) & set(top_k_generated)) / k
    recall = len(set(top_k_real) & set(top_k_generated)) / len(set(top_k_real) | set(top_k_generated))
    if presicion + recall == 0:
        score = 0
    else:
        score = 2 * presicion * recall / (presicion + recall)
    return score


def cosine_similarity(u, v):
    """Calculate cosine similarity between two vectors with protection against zero division."""
    epsilon = 1e-10
    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)

    if u_norm < epsilon or v_norm < epsilon:
        return 0
    return np.dot(u, v) / (u_norm * v_norm)


def main(args):
    dataset = args.city
    road_network = pd.read_csv(f'../traj_dataset/{dataset}/rn/edge_info.csv')

    gen_dataset_path = f'./logs/{dataset}/{args.exp_name}/data/generated_trajs_5e-4_layer1.pkl'
    gen_dataset = pload(gen_dataset_path)
    gen_road_trajs = [traj['road_traj'] for traj in gen_dataset]
    gen_gps_trajs = [np.round(np.vstack(d['gps_traj']), 6) for d in gen_dataset]

    real_dataset = pd.read_parquet(f'../traj_dataset/{dataset}/traj.parquet')
    test_index = np.load(f'../traj_dataset/{dataset}/test_index.npy')
    real_dataset = real_dataset.iloc[test_index].reset_index(drop=True)
    real_gps_trajs = [np.vstack(traj) for traj in real_dataset.geometry.tolist()]
    real_road_trajs = [traj for traj in real_dataset.opath.tolist()]

    real_road_density = eval_road_density(real_road_trajs, road_network)
    gen_road_density = eval_road_density(gen_road_trajs, road_network)
    real_gps2road_distances = eval_gps2road_distance(real_gps_trajs, real_road_trajs, road_network)
    gen_gps2road_distances = eval_gps2road_distance(gen_gps_trajs, gen_road_trajs, road_network)

    real_travel_distances, real_segment_distances = eval_travel_and_segment_distance(real_gps_trajs)
    gen_travel_distances, gen_segment_distances = eval_travel_and_segment_distance(gen_gps_trajs)
    real_radius = eval_radius(real_gps_trajs)
    gen_radius = eval_radius(gen_gps_trajs)

    real_density = eval_grid_density(real_gps_trajs, dataset, 100)
    gen_density = eval_grid_density(gen_gps_trajs, dataset, 100)

    js_travel_distance = calculate_jsd(real_travel_distances, gen_travel_distances)
    js_segment_distance = calculate_jsd(real_segment_distances, gen_segment_distances)
    js_radius = calculate_jsd(real_radius, gen_radius)

    grid_density = cosine_similarity(real_density, gen_density)
    grid_pattern_score = eval_pattern(real_density, gen_density, 100)

    road_density = cosine_similarity(real_road_density, gen_road_density)
    road_pattern_score = eval_pattern(real_road_density, gen_road_density, 100)
    js_gps2road_distance = calculate_jsd(real_gps2road_distances, gen_gps2road_distances)

    logger.add(sink=f'logs/{args.city}/{args.exp_name}/results.log', mode='w')
    logger.info(f"Travel-Distance: {'%.5f' % js_travel_distance}")
    logger.info(f"Segment-Distance: {'%.5f' % js_segment_distance}")
    logger.info(f"Radius: {'%.5f' % js_radius}")
    logger.info(f"Grid-Density: {'%.5f' % grid_density}")
    logger.info(f"Grid-Pattern: {'%.5f' % grid_pattern_score}")
    logger.info(f"Road-Density: {'%.5f' % road_density}")
    logger.info(f"Road-Pattern: {'%.5f' % road_pattern_score}")
    logger.info(f"Point2Road-Distance: {'%.5f' % js_gps2road_distance}")


if __name__ == '__main__':
    parser = HfArgumentParser(ParserArguments)
    args, = parser.parse_args_into_dataclasses()
    main(args)
