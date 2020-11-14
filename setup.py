from setuptools import setup

from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gradcache",
    version="0.0.0",
    description="A framework for weighted monte-carlo likelihood problems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/austinschneider/grad-cache",
    author="Austin Schneider",
    author_email="physics.schneider@gmail.com",
    license="L-GPL-3.0",
    packages=["gradcache", "gradcache/tests"],
    package_data={"gradcache": []},
    include_package_data=True,
    zip_safe=False,
)
