from setuptools import setup, find_packages

with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='coherentinfo',
    version='0.0.1',
    description='Python package for the calculation of coherent information in stabilizer codes',
    long_description=readme,
    author='Alessandro Ciani, Shobna Singh',
    author_email='alessandrociani89@gmail.com',
    url='https://github.com/cianibegood/stabwizard',
    license=license,
    packages=find_packages('.'),
    ext_package='coherentinfo',
    install_requires=list(open('requirements.txt').read().strip().split('\n'))
)