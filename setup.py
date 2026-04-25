from setuptools import setup, find_packages

setup(
    name="relational-calculus",
    version="0.1.0",
    description="Dimensionless Deep Learning framework for PyTorch. Escaping the Absolute Scale Trap.",
    author="Massimiliano Concas",
    author_email="tuo.indirizzo@email.com", # Inserisci la tua email se vuoi
    url="https://github.com/TuoUsername/relational-calculus", # Inserisci il link al tuo repo
    packages=find_packages(), # Trova in automatico la cartella "relational_calculus"
    install_requires=[
        "torch>=2.0.0", # L'unica vera dipendenza obbligatoria per il core
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
)
