from setuptools import setup

package_name = 'cone_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='debraj',
    maintainer_email='debraj@example.com',
    description='ROS2 package for cone detection using YOLO and CV Bridge',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cone_detector_node = cone_detector.cone_detector_node:main',
        ],
    },
)

