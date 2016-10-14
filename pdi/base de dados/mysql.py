__author__ = 'Wilson Junior'

import MySQLdb as mysql

conexao = mysql.connect(host="localhost",user="root",passwd="wilson",db="belezainovadora");

cursor = conexao.cursor();

cursor.execute("show tables");

resultado = cursor.fetchall();

for res in resultado:
    print(res);

