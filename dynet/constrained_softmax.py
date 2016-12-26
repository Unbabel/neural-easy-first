import numpy as np

def constrained_softmax(z, u):
    z -= np.mean(z)
    q = np.exp(z)
    active = np.ones(len(u))
    mass = 0.
    p = np.zeros(len(z))
    while True:
        inds = active.nonzero()[0]
        p[inds] = q[inds] * (1. - mass) / sum(q[inds])
        found = False
        #import pdb; pdb.set_trace()
        for i in inds:
            if p[i] > u[i]:
                p[i] = u[i]
                mass += u[i]
                found = True
                active[i] = 0
        if not found:
            break
    #print mass
    #print active
    return p, active, mass

def gradient_constrained_softmax(z, u, dp, p, active, mass):
    n = len(z)
    inds = active.nonzero()[0]
    Jz = np.zeros((n, n))
    Ju = np.zeros((n, n))
    for i in xrange(n):
        for j in xrange(n):
            if active[i]:
                if active[j]:
                    Jz[i, j] = -p[i] * p[j] / (1. - mass)
                else:
                    Ju[i, j] = -p[i] / (1. - mass)
        if active[i]:
            Jz[i, i] += p[i]
        else:
            Ju[i, i] = 1.

    #print 'Jz: ', Jz
    #print 'Ju: ', Ju
    dz = Jz.transpose().dot(dp)
    du = Ju.transpose().dot(dp)
    #import pdb; pdb.set_trace()
    return dz, du

def numeric_gradient_constrained_softmax(z, u, dp, p, active, mass):
    epsilon = 1e-6
    n = len(z)
    Jz = np.zeros((n, n))
    Ju = np.zeros((n, n))
    for j in xrange(n):
        z1 = z.copy()
        z2 = z.copy()
        z1[j] -= epsilon
        z2[j] += epsilon
        p1, _, _ = constrained_softmax(z1, u)
        p2, _, _ = constrained_softmax(z2, u)
        Jz[:, j] = (p2 - p1) / (2*epsilon)
        #import pdb; pdb.set_trace()

        u1 = u.copy()
        u2 = u.copy()
        u1[j] -= epsilon
        u2[j] += epsilon
        p1, _, _ = constrained_softmax(z, u1)
        p2, _, _ = constrained_softmax(z, u2)
        Ju[:, j] = (p2 - p1) / (2*epsilon)
    #print 'Jz_: ', Jz
    #print 'Ju_: ', Ju
    dz = Jz.transpose().dot(dp)
    du = Ju.transpose().dot(dp)
    #import pdb; pdb.set_trace()
    return dz, du

if __name__ == "__main__":
    n = 6
    z = np.random.randn(n)
    u = 0.5*np.random.rand(n)
    print sum(u)
    print z
    print u
    p, active, mass = constrained_softmax(z, u)
    print p
    print sum(p)

    dp = np.random.randn(n)
    dz, du = gradient_constrained_softmax(z, u, dp, p, active, mass)
    print dp
    print dz
    print du

    dz_, du_ = numeric_gradient_constrained_softmax(z, u, dp, p, active, mass)
    print dz_
    print du_

    print np.linalg.norm(dz - dz_)
    print np.linalg.norm(du - du_)
    
