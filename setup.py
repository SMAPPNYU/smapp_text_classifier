import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smapp_text_classifier",
    version="0.0.1.9000",
    author="Fridolin Linder, Michael Liu",
    author_email="fridolin.linder@nyu.edu",
    description="Helperfunctions for text classification for SMaPP@NYU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smappnyu/smapp_text_classifier",
    packages=setuptools.find_packages(),
    install_requires=[
            'spacy',
            'gensim',
            'scikit-learn',
            'pandas',
            'numpy'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
