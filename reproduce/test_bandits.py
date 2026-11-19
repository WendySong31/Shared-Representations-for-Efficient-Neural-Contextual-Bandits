import unittest
import numpy as np
from bandits import exp3_probs, update_networks, forward, neural_run, baseline_run
from prepare_data import encode

class BanditTests(unittest.TestCase):
    def test_exp3_probabilities_and_credit(self):
        weights=np.zeros(3)
        np.testing.assert_allclose(exp3_probs(weights,.1),np.ones(3)/3)
        weights[1]+=.1/(3*(1/3))
        probs=exp3_probs(weights,.1)
        self.assertAlmostEqual(probs.sum(),1.)
        self.assertGreater(probs[1],probs[0])
        self.assertTrue(np.all(probs>=.1/3))

    def test_feedback_isolation_and_gradient(self):
        rng=np.random.default_rng(12)
        x=np.array([1.,0.,1.])
        w1=rng.uniform(-.2,.2,(2,2,2,3));w2=rng.uniform(-.2,.2,(2,2,2))
        old1=w1.copy();old2=w2.copy()
        sizes=np.array([2,2]);rates=np.array([.01,.02])
        scores,hidden=forward(x,w1,w2,sizes)
        eps=1e-6
        plus=w1.copy();minus=w1.copy()
        plus[0,0,0,0]+=eps;minus[0,0,0,0]-=eps
        qp=forward(x,plus,w2,sizes)[0][0,0]
        qm=forward(x,minus,w2,sizes)[0][0,0]
        gradient=((1-qp)**2/2-(1-qm)**2/2)/(2*eps)
        update_networks(x,0,1.,.2,scores,hidden,w1,w2,sizes,rates)
        self.assertAlmostEqual(w1[0,0,0,0],old1[0,0,0,0]-.01/.2*gradient,places=9)
        np.testing.assert_array_equal(w1[:,1],old1[:,1])
        np.testing.assert_array_equal(w2[:,1],old2[:,1])
        self.assertTrue(np.all(np.any(w2[:,0]!=old2[:,0],axis=1)))

    def test_encoding(self):
        raw=np.zeros((10,55));raw[:,:10]=np.arange(10)[:,None];raw[:,-1]=1
        x,y,edges=encode(raw)
        self.assertEqual(x.shape,(10,94))
        np.testing.assert_array_equal(x[:,:50].sum(axis=1),np.full(10,10))
        np.testing.assert_array_equal(y,np.zeros(10))

    def test_stream_drift_and_determinism(self):
        x=np.ones((7,3),dtype=np.uint8);y=np.arange(7);order=np.arange(7)
        for variant in (1,2,3):
            sizes=np.array([2]) if variant==1 else np.array([1,2])
            rates=np.full(len(sizes),.1)
            args=(x,y,order,0,40,sizes,rates,variant,42,.005,.1,10)
            a=neural_run(*args);b=neural_run(*args)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertTrue(np.all(np.isfinite(a[1])))
            self.assertTrue(np.all((a[0]==0)|(a[0]==1)))
            if variant==2:self.assertEqual(a[2].sum(),40)
            if variant==3:self.assertEqual(a[2].sum(),280)
        for algorithm in (0,1,2):
            rewards=baseline_run(x,y,order,0,40,algorithm,42,drift_every=10)
            self.assertEqual(len(rewards),40)

    def test_committee_interpretations(self):
        x=np.ones((7,3),dtype=np.uint8);y=np.arange(7);order=np.arange(7)
        sizes=np.array([1,2]);rates=np.array([.01,.1])
        for variant,all_feedback,marginal in ((2,True,True),(3,False,False)):
            result=neural_run(x,y,order,0,40,sizes,rates,variant,42,
                              .005,.1,10,all_feedback,marginal)
            self.assertEqual(len(result[0]),40)
            self.assertTrue(np.all(np.isfinite(result[1])))

if __name__=='__main__':unittest.main()
