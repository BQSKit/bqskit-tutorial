OPENQASM 2.0;
include "qelib1.inc";
qreg node[9];
u2(0,pi) node[2];
u3(pi,0,pi) node[3];
u3(pi,0,pi) node[4];
u2(0,pi) node[6];
u3(pi,0,pi) node[7];
u3(pi,0,pi) node[8];
cx node[3],node[2];
cx node[4],node[8];
cx node[7],node[6];
u1(7*pi/4) node[2];
u1(7*pi/4) node[6];
cx node[1],node[2];
cx node[5],node[6];
u1(pi/4) node[2];
u1(pi/4) node[6];
cx node[3],node[2];
cx node[7],node[6];
u1(7*pi/4) node[2];
u1(pi/4) node[3];
u1(7*pi/4) node[6];
u1(pi/4) node[7];
cx node[1],node[2];
cx node[5],node[6];
u1(pi/4) node[2];
u1(pi/4) node[6];
cx node[1],node[2];
cx node[5],node[6];
cx node[2],node[3];
cx node[6],node[7];
cx node[1],node[2];
cx node[5],node[6];
u1(pi/4) node[1];
cx node[2],node[3];
u1(pi/4) node[5];
cx node[6],node[7];
cx node[0],node[1];
u1(7*pi/4) node[3];
u1(7*pi/4) node[7];
cx node[1],node[0];
cx node[3],node[2];
cx node[7],node[6];
cx node[0],node[1];
u1(7*pi/4) node[2];
u1(7*pi/4) node[6];
cx node[1],node[2];
u1(pi/4) node[2];
cx node[3],node[2];
u1(7*pi/4) node[2];
u1(pi/4) node[3];
cx node[1],node[2];
u2(0,5*pi/4) node[2];
cx node[2],node[6];
u1(pi/4) node[6];
cx node[7],node[6];
u1(7*pi/4) node[6];
u1(pi/4) node[7];
cx node[2],node[6];
cx node[2],node[3];
u2(0,5*pi/4) node[6];
cx node[3],node[2];
cx node[2],node[3];
cx node[1],node[2];
cx node[3],node[7];
u1(pi/4) node[1];
u1(7*pi/4) node[2];
u1(pi/4) node[3];
u1(7*pi/4) node[7];
cx node[1],node[2];
cx node[3],node[7];
cx node[8],node[7];
cx node[7],node[6];
cx node[8],node[7];
cx node[7],node[6];
u2(0,pi) node[6];
cx node[7],node[6];
u1(7*pi/4) node[6];
cx node[6],node[7];
cx node[7],node[6];
cx node[6],node[7];
cx node[3],node[7];
u1(pi/4) node[7];
cx node[6],node[7];
u1(pi/4) node[6];
u1(7*pi/4) node[7];
cx node[3],node[7];
cx node[2],node[3];
u1(pi/4) node[7];
cx node[3],node[2];
cx node[2],node[3];
cx node[2],node[6];
u1(pi/4) node[2];
u1(7*pi/4) node[6];
cx node[2],node[6];
cx node[5],node[6];
cx node[6],node[7];
cx node[5],node[6];
u1(7*pi/4) node[7];
cx node[6],node[7];
cx node[5],node[6];
cx node[6],node[7];
u1(pi/4) node[7];
cx node[6],node[7];
u1(pi/4) node[6];
u1(7*pi/4) node[7];
cx node[5],node[6];
cx node[6],node[5];
cx node[5],node[6];
cx node[6],node[7];
cx node[6],node[5];
u2(0,5*pi/4) node[7];
u1(7*pi/4) node[5];
u1(pi/4) node[6];
cx node[5],node[6];
cx node[6],node[5];
cx node[5],node[6];
cx node[2],node[6];
u2(0,pi) node[2];
cx node[3],node[2];
u1(7*pi/4) node[2];
cx node[1],node[2];
u1(pi/4) node[2];
cx node[3],node[2];
u1(7*pi/4) node[2];
u1(pi/4) node[3];
cx node[1],node[2];
u1(pi/4) node[2];
cx node[1],node[2];
cx node[2],node[3];
cx node[1],node[2];
u1(pi/4) node[1];
cx node[2],node[3];
cx node[1],node[2];
u1(7*pi/4) node[3];
cx node[2],node[3];
cx node[1],node[2];
cx node[0],node[1];
cx node[2],node[3];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[2];
cx node[2],node[3];
cx node[1],node[2];
cx node[2],node[3];
cx node[3],node[2];
u1(7*pi/4) node[2];
cx node[1],node[2];
u1(pi/4) node[2];
cx node[3],node[2];
u1(7*pi/4) node[2];
u1(pi/4) node[3];
cx node[1],node[2];
u2(0,5*pi/4) node[2];
cx node[2],node[3];
cx node[3],node[2];
cx node[2],node[3];
cx node[1],node[2];
u1(pi/4) node[1];
u1(7*pi/4) node[2];
cx node[0],node[1];
cx node[1],node[0];
cx node[0],node[1];
cx node[1],node[2];
