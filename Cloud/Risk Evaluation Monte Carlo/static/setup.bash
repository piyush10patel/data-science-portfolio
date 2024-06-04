"""#!/bin/bash
apt update -y
apt install python3 apache2 -y
apt install python3-pandas
apache2ctl restart
wget https://lateral-journey-377016.nw.r.appspot.com/cacheavoid/apache2.conf -P /var/www/html  #apache config files
wget https://lateral-journey-377016.nw.r.appspot.com/cacheavoid/getsignals.py -P /var/www/html
chmod 755 /var/www/html/apache2.config
chmod 755 /var/www/html/getsignals.py
a2enmod cgi
service apache2 restart
system ctl restart apache2
"""
