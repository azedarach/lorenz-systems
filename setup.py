from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(name="lorenz_systems",
          version="0.0.1",
          description="Example Lorenz-like systems",
          packages=find_packages(),
          install_requires=["numpy", "scipy"]
    )
