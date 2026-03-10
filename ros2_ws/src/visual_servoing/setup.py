from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'visual_servoing'

# Recursive model installation
def get_model_files():
    model_files = []
    for root, dirs, files in os.walk('models'):
        if files:
            # Get relative path from models/
            rel_path = os.path.relpath(root, 'models')
            if rel_path == '.':
                target_dir = os.path.join('share', package_name, 'models')
            else:
                target_dir = os.path.join('share', package_name, 'models', rel_path)
            
            file_paths = [os.path.join(root, f) for f in files]
            model_files.append((target_dir, file_paths))
    return model_files

# Recursive mesh installation  
def get_mesh_files():
    mesh_files = []
    for root, dirs, files in os.walk('meshes'):
        if files:
            # Get relative path from meshes/
            rel_path = os.path.relpath(root, 'meshes')
            if rel_path == '.':
                target_dir = os.path.join('share', package_name, 'meshes')
            else:
                target_dir = os.path.join('share', package_name, 'meshes', rel_path)
            
            file_paths = [os.path.join(root, f) for f in files]
            mesh_files.append((target_dir, file_paths))
    return mesh_files

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']) + ['drawing', 'rl', 'utils', 'agents'],
    package_dir={
        'drawing': 'scripts/drawing',
        'rl': 'scripts/rl',
        'utils': 'scripts/utils',
        'agents': 'scripts/agents',
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Install launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        # Install URDF files
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        # Install world files
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        # Install config files
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ] + get_model_files() + get_mesh_files(),  # Install models and meshes recursively
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ducanh',
    maintainer_email='do010303@gmail.com',
    description='Visual Servoing package for ceiling-mounted robot arm with ArUco markers',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'vision_aruco_detector = vs_lib.vision.vision_aruco_detector:main',
            'camera_viewer = vs_lib.vision.camera_viewer:main',
            'drawing_executor = vs_lib.nodes.drawing_executor_ros2:main',
            'shape_generator = vs_lib.nodes.shape_generator:main',
            'vision_node = vs_lib.nodes.vision_node_ros2:main',
            'gazebo_drawing_visualizer = drawing.gazebo_visualizer:main',
        ],
    },
)
