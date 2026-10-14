@echo off
echo Installing BioMorphX (v1.0.0) and required dependencies (numpy, scipy)...
python -m pip install --upgrade pip
python -m pip install numpy scipy
python -m pip install --force-reinstall dist\biomorphx-1.0.0-py3-none-any.whl
echo Running BioMorphX self-test...
python -m biomorphx.cli self-test
echo BioMorphX v1.0.0 is installed and ready to use!
pause
