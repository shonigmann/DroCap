## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    #version='...'
    #scripts=['bin/myscript']
    packages=['perch_placement','scripts','geom', 'sim'],
    package_dir={'': 'src'},
    requires=['rospy','open3d','numpy','Shapely','matplotlib','pyswarms','triangle','plyfile','pymesh','vedo','pyvista','trimesh','mpltools','mapbox_earcut'])

setup(**setup_args)
