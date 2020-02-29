sudo apt-get update

sudo apt-get install build-essential python-dev python-setuptools python-pip python-smbus

sudo apt-get install build-essential libncursesw5-dev libgdbm-dev libc6-dev

sudo apt-get install zlib1g-dev libsqlite3-dev tk-dev

sudo apt-get install libssl-dev openssl

sudo apt-get install libffi-dev

git clone git://github.com/yyuu/pyenv.git ~/.pyenv
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
exec $SHELL -l

pyenv install 3.7.0 -v
pyenv rehash
pyenv versions
pyenv global 3.7.0

pip3 install torch torchvision -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
