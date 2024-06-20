import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

long_description = "This is a project that takes a video as input and predict the fighting"

__version__ = "0.0.0"

REPO_NAME = "CCTV-Fight-Prediction-end-to-end-pipeline-dvc-aws-deploy"
AUTHOR_USER_NAME = "rohitw3code"
SRC_REPO = "fightClassifier"
AUTHOR_EMAIL = "rohitcode005@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for CNN app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)


