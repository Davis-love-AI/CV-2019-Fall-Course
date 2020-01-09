rm -rf build
rm rotate_utils.c
rm rotate_utils.cpython-35m-x86_64-linux-gnu.so
cp rotate_utils.py rotate_utils.pyx
python3 setup.py build_ext --inplace
rm ../rotate_utils.cpython-37m-x86_64-linux-gnu.so
cp rotate_utils.cpython-37m-x86_64-linux-gnu.so ../
