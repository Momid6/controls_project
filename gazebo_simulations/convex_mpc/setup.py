from setuptools import find_packages, setup

package_name = 'convex_mpc'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mohammed-dawood',
    maintainer_email='mohammed-dawood@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'convex_mpc_node = convex_mpc.convex_mpc_node:main',
            'convex_mpc_with_tracking_v2 = convex_mpc.convex_mpc_with_tracking_v2:main',
            'convex_mpc_with_tracking_v1 = convex_mpc.convex_mpc_with_tracking_v1:main',
            'convex_mpc_advanced_node = convex_mpc.convex_mpc_advanced_node:main',
        ],
    },
)
