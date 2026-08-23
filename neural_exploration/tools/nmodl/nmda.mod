NEURON {
	POINT_PROCESS NMDASyn
	RANGE tau, e, gmax, mg
	NONSPECIFIC_CURRENT i
}
UNITS {
	(nA) = (nanoamp)
	(mV) = (millivolt)
	(uS) = (microsiemens)
	(mM) = (milli/liter)
}
PARAMETER {
	tau = 100 (ms)
	e = 0 (mV)
	gmax = 1 (uS)
	mg = 1.2 (mM)
}
ASSIGNED {
	v (mV)
	i (nA)
	B (1)
}
STATE {
	g (uS)
}
INITIAL {
	g = 0
}
BREAKPOINT {
	SOLVE state METHOD cnexp
	i = g*(v - e)
}
DERIVATIVE state {
	g' = -g/tau
}
NET_RECEIVE(weight (uS)) {
	B = 1/(1 + mg*exp(-0.062*v)/3.57)
	g = g + weight*B
}
