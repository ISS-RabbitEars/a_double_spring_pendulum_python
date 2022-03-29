import numpy as np
import sympy as sp
from sympy.physics.vector import dynamicsymbols
from scipy.integrate import odeint
from matplotlib import pyplot as plt
from matplotlib import animation

def integrate(ic, ti, p):
	m1, m2, k1, k2, r1eq, r2eq, gc = p
	r1, v1, theta1, omega1, r2, v2, theta2, omega2 = ic

	print(ti)

	return[v1,A1.subs({M1:m1, K1:k1, K2:k2, R1:r1, R2:r2, R1EQ:r1eq, R2EQ:r2eq, THETA1:theta1, THETA2:theta2, THETA1dot:omega1, g:gc}),\
		omega1,ALPHA1.subs({M1:m1, R1:r1, R2:r2, R1dot:v1, K2:k2, R2EQ:r2eq, THETA1:theta1, THETA2:theta2, THETA1dot:omega1, g:gc}),\
		v2,A2.subs({M1:m1, M2:m2, K1:k1, K2:k2, R1EQ:r1eq, R2EQ:r2eq, R1:r1, R2:r2, THETA1:theta1, THETA2:theta2, THETA2dot:omega2}),\
		omega2,ALPHA2.subs({M1:m1, K1:k1, R1EQ:r1eq, R1:r1, R2:r2, R2dot:v2, THETA1:theta1, THETA2:theta2, THETA2dot:omega2})]


M1, M2, K1, K2, R1EQ, R2EQ, g, t = sp.symbols('M1 M2 K1 K2 R1EQ R2EQ g t')
R1, R2, THETA1, THETA2 = dynamicsymbols('R1 R2 THETA1 THETA2')

X1 = R1 * sp.sin(THETA1)
Y1 = - R1 * sp.cos(THETA1)
X2 = X1 + R2 * sp.sin(THETA2)
Y2 = Y1 - R2 * sp.cos(THETA2)

X1dot = X1.diff(t, 1)
Y1dot = Y1.diff(t, 1)
X2dot = X2.diff(t, 1)
Y2dot = Y2.diff(t, 1)

V1s = X1dot**2 + Y1dot**2
V2s = X2dot**2 + Y2dot**2

T = sp.Rational(1, 2) * (M1 * V1s + M2 * V2s)
T = sp.simplify(T)
V = sp.Rational(1, 2) * (K1 * (R1 - R1EQ)**2 + K2 * (R2 - R2EQ)**2) + g * (M1 * Y1 + M2 * Y2)
V = sp.simplify(V)

L = T - V

R1dot = R1.diff(t, 1)
dLdR1 = L.diff(R1, 1)
dLdR1dot = L.diff(R1dot, 1)
ddtdLdR1dot = dLdR1dot.diff(t, 1)
dLR1 = ddtdLdR1dot - dLdR1

R2dot = R2.diff(t, 1)
dLdR2 = L.diff(R2, 1)
dLdR2dot = L.diff(R2dot, 1)
ddtdLdR2dot = dLdR2dot.diff(t, 1)
dLR2 = ddtdLdR2dot - dLdR2

THETA1dot = THETA1.diff(t, 1)
dLdTHETA1 = L.diff(THETA1, 1)
dLdTHETA1dot = L.diff(THETA1dot, 1)
ddtdLdTHETA1dot = dLdTHETA1dot.diff(t, 1)
dLTHETA1 = ddtdLdTHETA1dot - dLdTHETA1

THETA2dot = THETA2.diff(t, 1)
dLdTHETA2 = L.diff(THETA2, 1)
dLdTHETA2dot = L.diff(THETA2dot, 1)
ddtdLdTHETA2dot = dLdTHETA2dot.diff(t, 1)
dLTHETA2 = ddtdLdTHETA2dot - dLdTHETA2

R1ddot = R1.diff(t, 2)
R2ddot = R2.diff(t, 2)
THETA1ddot = THETA1.diff(t, 2)
THETA2ddot = THETA2.diff(t, 2)
sol = sp.solve([dLR1, dLR2, dLTHETA1, dLTHETA2], (R1ddot, R2ddot, THETA1ddot, THETA2ddot))

A1 = sp.simplify(sol[R1ddot])
ALPHA1 = sp.simplify(sol[THETA1ddot])
A2 = sp.simplify(sol[R2ddot])
ALPHA2 = sp.simplify(sol[THETA2ddot])

#-----------------------------------------------

gc = 9.8
m1, m2 = [1, 1]
k1, k2 = [10, 10] 
r1eq, r2eq = [1, 1]
r1o, r2o = [2, 2]
v1o, v2o = [0, 0]
theta1o, theta2o = [90, 180] 
omega1o, omega2o = [0, 0]
cnvrt = np.pi/180
theta1o *= cnvrt
omega1o *= cnvrt
theta2o *= cnvrt
omega2o *= cnvrt


tf = 60 
nfps = 30
nframes = tf * nfps
ta = np.linspace(0, tf, nframes)


p = m1, m2, k1, k2, r1eq, r2eq, gc
ic = r1o, v1o, theta1o, omega1o, r2o, v2o, theta2o, omega2o


rth = odeint(integrate, ic, ta, args=(p,))

x1 = np.asarray([X1.subs({R1:rth[i,0], THETA1:rth[i,2]}) for i in range(nframes)])
y1 = np.asarray([Y1.subs({R1:rth[i,0], THETA1:rth[i,2]}) for i in range(nframes)])
x2 = np.asarray([X2.subs({X1:x1[i], R2:rth[i,4], THETA2:rth[i,6]}) for i in range(nframes)])
y2 = np.asarray([Y2.subs({Y1:y1[i], R2:rth[i,4], THETA2:rth[i,6]}) for i in range(nframes)])

ke = np.asarray([T.subs({M1:m1, M2:m2, R1:rth[i,0], R1dot:rth[i,1], THETA1:rth[i,2], THETA1dot:rth[i,3],\
	R2:rth[i,4], R2dot:rth[i,5], THETA2:rth[i,6], THETA2dot:rth[i,7]}) for i in range(nframes)])
pe = np.asarray([V.subs({M1:m1, M2:m2, K1:k1, K2:k2, R1EQ:r1eq, R2EQ:r2eq, g:gc,\
	R1:rth[i,0], THETA1:rth[i,2], R2:rth[i,4], THETA2:rth[i,6]}) for i in range(nframes)])
E = ke + pe

#---------------------------------------------------

if m1 > m2:
	rad1 = 0.25
	rad2 = (m2/m1) * rad1
else:
	rad2 = 0.25
	rad1 = (m1/m2) * rad2

xmax = max(x1) + 2 * rad1 if max(x1) > max(x2) else max(x2) + 2 * rad2
xmin = min(x1) - 2 * rad1 if min(x1) < min(x2) else min(x2) - 2 * rad2
ymax = max(y1) + 2 * rad1 if max(y1) > max(y2) else max(y2) + 2 * rad2
if ymax < 0 : ymax = 0 + rad1 + rad2
ymin = min(y1) - 2 * rad1 if min(y1) < min(y2) else min(y2) - 2 * rad2

r1max = max(np.abs(rth[:,0]))
r2max = max(np.abs(rth[:,4]))
nl1 = int(np.ceil(r1max/(2*rad1)))
nl2 = int(np.ceil(r2max/(2*rad2)))
l1 = (np.asarray(np.abs(rth[:,0]))-rad1)/nl1
l2 = (np.asarray(np.abs(rth[:,4]))-(rad2+rad1))/nl2
h1 = np.sqrt(rad1**2 - (0.5*l1)**2)
h2 = np.sqrt(rad2**2 - (0.5*l2)**2)
xl1o = x1 - rad1*np.cos(np.pi/2 - np.asarray(rth[:,2]))
yl1o = y1 + rad1*np.sin(np.pi/2 - np.asarray(rth[:,2]))
xl2o = x2 - rad2*np.cos(np.pi/2 - np.asarray(rth[:,6]))
yl2o = y2 + rad2*np.sin(np.pi/2 - np.asarray(rth[:,6]))
x2pp = x1 + rad1*np.sin(np.asarray(rth[:,2]))
y2pp = y1 - rad1*np.cos(np.asarray(rth[:,2]))
xl1 = np.zeros((nl1,nframes))
yl1 = np.zeros((nl1,nframes))
xl2 = np.zeros((nl2,nframes))
yl2 = np.zeros((nl2,nframes))
for i in range(nframes):
	xl1[0][i] = xl1o[i] - 0.5 * l1[i] * np.cos(np.pi/2 - rth[i,2]) + h1[i] * np.cos(rth[i,2])
	yl1[0][i] = yl1o[i] + 0.5 * l1[i] * np.sin(np.pi/2 - rth[i,2]) + h1[i] * np.sin(rth[i,2])
for j in range(nframes):
	for i in range(1,nl1):
		xl1[i][j] = xl1o[j] - (0.5 + i) * l1[j] * np.cos(np.pi/2 - rth[j,2]) + (-1)**i * h1[j] * np.cos(rth[j,2])
		yl1[i][j] = yl1o[j] + (0.5 + i) * l1[j] * np.sin(np.pi/2 - rth[j,2]) + (-1)**i * h1[j] * np.sin(rth[j,2])
for i in range(nframes):
	xl2[0][i] = xl2o[i] - 0.5 * l2[i] * np.cos(np.pi/2 - rth[i,6]) + h2[i] * np.cos(rth[i,6])
	yl2[0][i] = yl2o[i] + 0.5 * l2[i] * np.sin(np.pi/2 - rth[i,6]) + h2[i] * np.sin(rth[i,6])
for j in range(nframes):
	for i in range(1,nl2):
		xl2[i][j] = xl2o[j] - (0.5 + i) * l2[j] * np.cos(np.pi/2 - rth[j,6]) + (-1)**i * h2[j] * np.cos(rth[j,6])
		yl2[i][j] = yl2o[j] + (0.5 + i) * l2[j] * np.sin(np.pi/2 - rth[j,6]) + (-1)**i * h2[j] * np.sin(rth[j,6])

fig, a=plt.subplots()

def run(frame):
	plt.clf()
	plt.subplot(211)
	circle=plt.Circle((x1[frame],y1[frame]),radius=rad1,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([xl1o[frame],xl1[0][frame]],[yl1o[frame],yl1[0][frame]],'xkcd:cerulean')
	plt.plot([xl1[nl1-1][frame],0],[yl1[nl1-1][frame],0],'xkcd:cerulean')
	for i in range(nl1-1):
		plt.plot([xl1[i][frame],xl1[i+1][frame]],[yl1[i][frame],yl1[i+1][frame]],'xkcd:cerulean')
	circle=plt.Circle((x2[frame],y2[frame]),radius=rad2,fc='xkcd:red')
	plt.gca().add_patch(circle)
	plt.plot([xl2o[frame],xl2[0][frame]],[yl2o[frame],yl2[0][frame]],'xkcd:cerulean')
	plt.plot([xl2[nl2-1][frame],x2pp[frame]],[yl2[nl2-1][frame],y2pp[frame]],'xkcd:cerulean')
	for i in range(nl2-1):
		plt.plot([xl2[i][frame],xl2[i+1][frame]],[yl2[i][frame],yl2[i+1][frame]],'xkcd:cerulean')
	plt.title("A Double Spring Pendulum")
	ax=plt.gca()
	ax.set_aspect(1)
	plt.xlim([float(xmin),float(xmax)])
	plt.ylim([float(ymin),float(ymax)])
	ax.xaxis.set_ticklabels([])
	ax.yaxis.set_ticklabels([])
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_facecolor('xkcd:black')
	plt.subplot(212)
	plt.plot(ta[0:frame],ke[0:frame],'xkcd:red',lw=1.0)
	plt.plot(ta[0:frame],pe[0:frame],'xkcd:cerulean',lw=1.0)
	plt.plot(ta[0:frame],E[0:frame],'xkcd:bright green',lw=1.5)
	plt.xlim([0,tf])
	plt.title("Energy")
	ax=plt.gca()
	ax.legend(['T','V','E'],labelcolor='w',frameon=False)
	ax.set_facecolor('xkcd:black')

ani=animation.FuncAnimation(fig,run,frames=nframes)
writervideo = animation.FFMpegWriter(fps=nfps)
ani.save('double_spring_pendulum.mp4', writer=writervideo)
plt.show()



