import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'beverage_disposal_bringup'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='rouna',
    maintainer_email='rounakrai.mail@gmail.com',
    description='Fortress/gz-sim bringup for the beverage disposal arm simulation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ultrasonic_range_node = beverage_disposal_bringup.ultrasonic_range_node:main',
            'pick_and_place_demo = beverage_disposal_bringup.pick_and_place_demo:main',
        ],
    },
)
