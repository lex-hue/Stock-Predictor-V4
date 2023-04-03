echo "Cancel the script if you didnt setup the venv yet (Optional)"
sleep 2
clear
echo "If you did or dont want an venv setup wait 3 secs"
sleep 3
wget https://deac-fra.dl.sourceforge.net/project/ta-lib/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xvzf ta-lib-0.4.0-src.tar.gz
cd ta-lib
./configure --prefix=/usr
make
sudo make install
cd ..
pip install -r requirements.txt
