import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

reqs = []
setuptools.setup(
    name='dp4inversion',
    version='0.2.0',
    author='Charlan D. S. Alves',
    author_email='cdellon@lamce.coppe.ufrj.br',
    description='Deep Bayesian inference for full waveform inversion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Uncertainty-Quantification-Project/deepprior2inversion',
    license='MIT',
    install_requires=reqs,
    python_requires='>=3.10',
    packages=setuptools.find_packages(include=['dp4inversion*']),
)
