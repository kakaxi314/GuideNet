from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GuideConv',
    ext_modules=[
        CUDAExtension('GuideConv', [
            'guideconv.cpp',
            'guideconv_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })