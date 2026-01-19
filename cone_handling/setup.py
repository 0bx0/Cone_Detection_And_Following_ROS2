from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cone_handling'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'action'), glob('action/*.action')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='d4y0n3',
    maintainer_email='f20230411@goa.bits-pilani.ac.in',
    description='Cone detection and following package with action server support',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cone_handling_node = cone_handling.cone_handling_node:main',
            'cone_detector_node = cone_handling.cone_detector_node:main',
            'cone_follower_node = cone_handling.cone_follower_node:main',
            'cone_follower_action_server = cone_handling.cone_follower_action_server:main',
        ],
    },
)
