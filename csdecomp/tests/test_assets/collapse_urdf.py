import xml.etree.ElementTree as ET
import numpy as np

def parse_origin(origin_element):
    if origin_element is None:
        return np.eye(4)
    
    xyz = [float(x) for x in origin_element.get('xyz', '0 0 0').split()]
    rpy = [float(r) for r in origin_element.get('rpy', '0 0 0').split()]
    
    # Create transformation matrix
    T = np.eye(4)
    T[:3, 3] = xyz
    
    # Convert roll, pitch, yaw to rotation matrix
    cx, cy, cz = np.cos(rpy)
    sx, sy, sz = np.sin(rpy)
    R = np.array([
        [cz*cy, cz*sy*sx - sz*cx, cz*sy*cx + sz*sx],
        [sz*cy, sz*sy*sx + cz*cx, sz*sy*cx - cz*sx],
        [-sy, cy*sx, cy*cx]
    ])
    T[:3, :3] = R
    
    return T

def combine_transformations(T1, T2):
    return np.dot(T1, T2)

def transformation_to_xyz_rpy(T):
    xyz = T[:3, 3]
    
    # Extract roll, pitch, yaw from rotation matrix
    sy = np.sqrt(T[0, 0] * T[0, 0] + T[1, 0] * T[1, 0])
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(T[2, 1], T[2, 2])
        pitch = np.arctan2(-T[2, 0], sy)
        yaw = np.arctan2(T[1, 0], T[0, 0])
    else:
        roll = np.arctan2(-T[1, 2], T[1, 1])
        pitch = np.arctan2(-T[2, 0], sy)
        yaw = 0
    
    return xyz, [roll, pitch, yaw]

def collapse_fixed_joints(urdf_file):
    tree = ET.parse(urdf_file)
    root = tree.getroot()
    
    # Find the first link (root link)
    root_link = root.find('link')
    root_link_name = root_link.get('name')
    
    # Create a dictionary to store link transforms
    link_transforms = {root_link_name: np.eye(4)}
    
    # Create a dictionary to map child links to their parent joints
    child_to_parent = {}
    for joint in root.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        child_to_parent[child] = (parent, joint)
    
    # Function to recursively compute transforms
    def compute_transform(link_name):
        if link_name in link_transforms:
            return link_transforms[link_name]
        
        parent_name, joint = child_to_parent[link_name]
        parent_transform = compute_transform(parent_name)
        joint_transform = parse_origin(joint.find('origin'))
        link_transform = combine_transformations(parent_transform, joint_transform)
        link_transforms[link_name] = link_transform
        return link_transform
    
    # Compute transforms for all links
    for link in root.findall('link'):
        compute_transform(link.get('name'))
    
    # Initialize combined visual and collision elements
    combined_visual = []
    combined_collision = []
    
    # Process all links except the root link
    for link in root.findall('link'):
        if link.get('name') != root_link_name:
            link_transform = link_transforms[link.get('name')]
            
            # Process visual elements
            for visual in link.findall('visual'):
                visual_origin = visual.find('origin')
                visual_transform = combine_transformations(link_transform, parse_origin(visual_origin))
                xyz, rpy = transformation_to_xyz_rpy(visual_transform)
                
                if visual_origin is None:
                    visual_origin = ET.SubElement(visual, 'origin')
                
                visual_origin.set('xyz', f"{xyz[0]} {xyz[1]} {xyz[2]}")
                visual_origin.set('rpy', f"{rpy[0]} {rpy[1]} {rpy[2]}")
                combined_visual.append(visual)
            
            # Process collision elements
            for collision in link.findall('collision'):
                collision_origin = collision.find('origin')
                collision_transform = combine_transformations(link_transform, parse_origin(collision_origin))
                xyz, rpy = transformation_to_xyz_rpy(collision_transform)
                
                if collision_origin is None:
                    collision_origin = ET.SubElement(collision, 'origin')
                
                collision_origin.set('xyz', f"{xyz[0]} {xyz[1]} {xyz[2]}")
                collision_origin.set('rpy', f"{rpy[0]} {rpy[1]} {rpy[2]}")
                combined_collision.append(collision)
            
            # Remove processed link
            root.remove(link)
    
    # Add combined visual and collision elements to root link
    for visual in combined_visual:
        root_link.append(visual)
    for collision in combined_collision:
        root_link.append(collision)
    
    # Remove all joints
    for joint in root.findall('joint'):
        root.remove(joint)
    
    # Write the modified URDF to a new file
    tree.write('collapsed_robotiq.urdf', encoding='utf-8', xml_declaration=True)

# Usage
collapse_fixed_joints('simple_robotiq.urdf')