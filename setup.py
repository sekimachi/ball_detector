from setuptools import find_packages, setup
import os
from setuptools import setup
import os
from glob import glob

package_name = 'ball_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('lib', 'python3.10', 'site-packages', package_name),
            ['ball_detector/best_blue.pt']
        ),
        (os.path.join('lib', 'python3.10', 'site-packages', package_name),
            ['ball_detector/best_red.pt']
        ),
        (os.path.join('lib', 'python3.10', 'site-packages', package_name),
            ['ball_detector/best_yellow.pt']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='imamura',
    maintainer_email='sekimachi.287@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'ball_detector = ball_detector.ball_detector_node:main',
        ],
    },
)
