from setuptools import setup, find_packages

setup(
    name="icu_analysis",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'scikit-learn==1.0.2',
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0',
        'shap>=0.40.0',
        'lime>=0.2.0.1',
        'fairlearn>=0.7.0'
    ],
    python_requires='>=3.10',
) 