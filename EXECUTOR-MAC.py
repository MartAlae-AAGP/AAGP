#%%


### - START PYTHON
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "pyDeepGP"))

import os, venv, subprocess, platform
def RUN_PIPELINE(test_function = 0):
    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - SETTINGS
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """

    import subprocess
    import sys
    import os
    import gc
    '''
    import EXAMPLE_FUNCTION
    '''
    # if True:
    def install_packages(skip_tokens=['git']):
        def install(package):
            print('Installing package: ', package)
            if not 'requirements' in package.lower():
                if 'matplot' in package or 'scipy' in package or 'xgboost' in package or 'pandas' in package or 'scikit-learn' in package:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    except:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package.split('=')[0]])
                else:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                    except:
                        subprocess.check_call([sys.executable, "-m", "pip", "install",'--use-pep517', package])
            else:
                subprocess.check_call([sys.executable,"-m", "pip", "install", '-r', package])

        packages = [
            # 'matplotlib==3.7.1',
            # 'seaborn==0.12.2',
            # 'tqdm==4.65.0',
            # 'scipy==1.8.0', # 1.7.3 for python 3.7.16
            # 'scikit-learn==1.0.2',
            # 'xgboost==1.7.5', # 1.6.2 for python 3.7.16
            # 'GPy==1.10.0',
            # 'git+https://github.com/SheffieldML/pyDeepGP',
            # 'numpy==1.21.6',
            # 'pandas==2.0.1',
            # 'joblib==1.2.0',
            # 'ipykernel',
            'numpy>=1.23.1',
            '../requirements.txt'
        ]
        print()
        print('================================================================')
        print('[ SYSTEM INFO ] - PYTHON VERSION: < %s > '%('.'.join([str(g) for g in sys.version_info[:3]])))
        print('================================================================')
        print()
        for g in packages:
            if 'requir' in g:
                with open(g) as file:
                    packs = [_ for _ in file.read().split('\n') if len(_) > 0 ]
                    if skip_tokens:
                        skips = [g_ for g_ in packs if True in [f_ in g_ for f_ in skip_tokens]]
                        if len(skips) > 0:
                            print(f'Skipping packages: {skips}')
                        packs = [g for g in packs if not g in skips]
                        print(f'Installing packages: {packs}')
                    # for _g in packs:
                    #     install(_g)
                    with open('temp_requirements.txt','w', encoding='utf-8') as tempfile:
                        for pkg in packs:
                            tempfile.write(pkg+'\n') 
                    install('temp_requirements.txt')
                    continue
            else:
                install(g)
        print()
        print()
        print()
        print()
        print()
        print()

    install_packages()
    test_function   = 'z.5.3' if test_function == 0 else 'z.1.10'
    deepGP_maxIters = 2000 if not test_function == 'z.5.3' else 200
    inits           = [4,8,10]
    pops            = [200,350,500]
    n_find          = 30
    noise           = 1 # percent value, do not convert to decimal.
    parallel        = True
    n_jobs          = int(min(os.cpu_count()-4, 32))
    if n_jobs <=0:
        n_jobs = 1
    models          = ['gp','aagp','localridge-gbal','xgboost-gbal','lod','lrk','slrgp','deepgp']
    replicates      = 20

    renames = {
        'gp':'GP (ALM)',
        'aagp':'AAGP (Adj.Ada. ALM)',
        'localridge-gbal':'LocalRidge (GBAL)',
        'xgboost-gbal':'XGBoost (GBAL)',
        'lod':'LOD (Lap.Reg. DoD)',
        'lrk':'LRK (Lap.Reg. ALM)',
        'slrgp':'SLRGP (Lap.Reg. ALC)',
        'deepgp':'DeepGP (Deep ALM)',
    }

    model_palette   = {
        'GP (ALM)': '#30a2da',
        'AAGP (Adj.Ada. ALM)': '#fc4f30',
        'LocalRidge (GBAL)': '#e5ae38',
        'XGBoost (GBAL)': '#6d904f',
        'XGBoost (GBAL)': '#6d904f',
        'LOD (Lap.Reg. DoD)': '#8b8b8b',
        'LRK (Lap.Reg. ALM)': '#e630e6',
        'SLRGP (Lap.Reg. ALC)': '#30309c',
        'DeepGP (Deep ALM)': '#423030',
    }







    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - AAGP
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """



    # [1] - package import
    # =========================
    import numpy as np
    import matplotlib.pyplot as plt
    # from scipy.stats import gaussian_kde
    # from matplotlib import cm


    # [2] - self-made package import
    # ===============================
    '''
    from algorithms import(
        euclidean_distances,
        get_kernel_data_bounds as get_kernel_data_bounds,
        myKernel,
    )

    from optimization import(
        class_execute_optimizer as class_execute_optimizer
    )

    from visualization import(
        dawn_cmap
    )
    '''


    def euclidean_correction(xRef, xTarget, dRef=None, dTarget=None):

        '''
        # Euclidean Correction
        Generates a coefficient to ensure that `xRef` and `xTarget` are on the same euclidean scale.
        '''
        ff = lambda xin: euclidean_distances(xin)[np.tril_indices(xin.shape[0],k=-1)]

        # [1] - get the reference frame
        if dRef is None:
            uSig= ff(xRef).max()
        else:
            uSig = dRef

        # [2] - get the target frame
        if dTarget is None:
            xSig= ff(xTarget).max()
        else:
            xSig = dTarget

        if uSig==0:
            uSig=1
        if xSig==0:
            xSig=1


        # [3] - make the transforming values.
        coeff= np.divide(xSig,uSig)

        return coeff

    def calculate_ALE(x, y,):
        '''
        # Analytical Likelihood calculation (ALE)
        This function calculates the optimal lengthscale given data-driven parameters for signal and noise variance using geostatistics.
        '''

        # [1] - get the pairwise distances
        # ==================================
        d = euclidean_distances(x,x)
        v = euclidean_distances(y,y)
        idx = np.where(v>0)

        dt = d[idx]
        vt = v[idx]

        idx = np.argsort(dt)
        vt  = vt[idx]
        dt  = dt[idx]

        # [2] - set the data-driven values
        # ====================================
        valid = lambda xIn: np.mean(xIn[np.isfinite(xIn)])
        n     = vt.min()
        vg    = vt-n
        s     =  np.mean(vg)/2 + np.percentile(vg,50)/2

        # [3] - calculate the optimal lengthscale
        # ========================================
        c = -1
        G = 2 * s / (2 * (n + s) - vg) + 2j * np.pi * c
        R = np.divide(dt,G)
        R = np.sqrt(np.square(R.real) + np.square(R.imag))
        R = valid(R)
        
        return s,n,R





    class AdjacencyVectorizer:

        def __init__(self, x, y, xa=None, train=True, plot_ALE=False):

            self.x     = x
            self.y     = y
            self.xa    = xa if not xa is None else x
            self.train = train
            self.prepare_training_data( plot_ALE=plot_ALE )
        
        def create_kernel_matrix(self, x1,x2,g=None, r = 1):
            # print(g)
            # if g is None:
            #     r = np.ravel([float(np.abs(r))] * x1.shape[1])
            #     x1= np.divide(x1,r)
            #     x2= np.divide(x2,r)
            #     g = euclidean_distances(x1,x2)
            
            # K = np.exp(-0.5 * np.square(g))
            K = myKernel(x1,x2,g=g,r=r,s=1,ARD=False,kernel='gaussian')
            # print(K)
            return K

        
        def prepare_training_data(self, plot_ALE=False):

            # this function will prepare the training data for speed optimization when running training.
            x,y,xa = self.x, self.y, self.xa
            d = euclidean_distances(x,x)
            da= euclidean_distances(xa,xa)
            v = euclidean_distances(y,y)

            tril_s = np.tril_indices(x.shape[0],k=-1)
            tril_a = np.tril_indices(xa.shape[0],k=-1)

            dt, vt = [g[tril_s] for g in [d,v]]
            dat    = da[tril_a]


            sIdx = []
            for i in range(x.shape[0]):
                xi = x[i,:]
                delta = np.ravel(np.abs(xi-xa).sum(1))
                idx = np.argmin(delta)
                if delta[idx] == 0:
                    sIdx.append(i)
            uIdx = [g for g in range(xa.shape[0]) if not g in sIdx]


            self.d = d
            self.da= da
            self.v = v

            self.dt  = dt
            self.dat = dat
            self.vt  = vt

            self.tril_s = tril_s
            self.tril_a = tril_a

            # [1] - grab the bounds for the training.
            # =======================================================================
            # trils, trila, dsx,v, dax, dst, vt, dat, sBounds,nBounds,rBounds,rBounds = get_kernel_data_bounds(x,y,xa=xa)
            idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds  = get_kernel_data_bounds(x,y,xa=xa)
            self.rBounds = rBounds
            self.dt = dt
            self.dat= dat
            
            # [1.1] - get a correction coefficient and function (REQUIRED)
            # =======================================================================
            fc = euclidean_correction(xa,xTarget=xa,dTarget=dat.max())
            self.Z_X = lambda xIn: xIn * fc
            zxa = self.Z_X(xa)
            zxs = self.Z_X(x)
            self.zxs = zxs
            self.zxa = zxa


            # [2] - calculate the by-dimension training distances.
            # =======================================================================
            VECS = []
            VECSA= []
            for i in range(x.shape[1]):

                da_ = euclidean_distances(zxa[:,i].reshape(-1,1), zxa[:,i].reshape(-1,1))
                VECSA.append(da_)
                # d_ = euclidean_distances(zxs[:,i].reshape(-1,1), zxa[:,i].reshape(-1,1))
                VECS.append(da_[sIdx,:])


            
            self.VECS  = VECS
            self.VECSA = VECSA

            # [3] - Utilize analytical likelihood estimation (ALE)
            # =======================================================================
            rho_s, rho_n, rho_r    = calculate_ALE(x,y)
                
            self.rho_s = rho_s
            self.rho_n = rho_n
            self.rho_r = rho_r
        
        def vectorize(self, xIn,  r=1, mode=['train','test'][0], use_all = False):

            if mode=='train':
                V = self.VECS
                if use_all:
                    V = self.VECSA
                
                J = np.column_stack([self.create_kernel_matrix(None,None, g=f/float(r), r=1).mean(1).reshape(-1,1) for f in V])

            else:
                zIn = self.Z_X(xIn)
                J = np.column_stack([self.create_kernel_matrix(zIn[:,i].reshape(-1,1), self.zxa[:,i].reshape(-1,1), r=r).mean(1).reshape(-1,1) for i in range(xIn.shape[1])])
            
            return J
        

        def loss_function(self, xIn, hypersIn, mode=['train','test'][0]):

            r = np.ravel(hypersIn).tolist()[0]
            j = self.vectorize(None, r=r, mode='train', use_all = not 'train' in mode)
            
            
            if mode=='test':
                # varxi = self.dat.max()/euclidean_distances(j,j).max()
                varxi = euclidean_correction(xRef = j, xTarget = self.xa, dTarget = self.dat.max())
                SV = lambda xIn: self.vectorize(xIn, r=r, mode='test') * varxi
                return SV
            
            else:
                # varxi = self.dt.max()/euclidean_distances(j,j).max()
                varxi = euclidean_correction(xRef = j, xTarget = self.x, dTarget = self.dt.max())
                J = j * varxi
                K = self.rho_s * (1 - self.create_kernel_matrix(J,J,r=self.rho_r)) + self.rho_n
                K = K[self.tril_s]

                m = np.log(K) + np.divide(self.vt, 2.0 * K)
                m = m[np.isfinite(m)].sum()
                return m
        
        def fit(self):

            optLo     = [self.dt[self.dt>0].min()]
            optHi     = [self.dat.max()]
            optBounds = [optLo,optHi]
            optFunc   = lambda hypersIn: self.loss_function(None, hypersIn, mode='train')
            retFunc   = lambda hypersIn: self.loss_function(None, hypersIn, mode='test')
            if self.train:
                # xOpt = lhs_optimization(optFunc, optBounds, n_points=30)
                _, xOpt,yOpt = class_execute_optimizer(
                    optFunc = optFunc,
                    optBounds = optBounds,
                    addCorners = False,
                    O = 'pso',

                )
                SV = retFunc(xOpt)
            else:
                SV = retFunc(np.ravel(np.matrix(optBounds).mean(0)))
            self.SV = SV
            
            return SV
        
        def plot(self, gsize=50, dpi = 100, figsize=(4,4), ax=None):
            
            # this will plot a 2D function of the kernel at the first measured point.
            x  = self.x
            xs = self.x[0,:].reshape(1,-1)
            rho_r = self.rho_r
            SV    = self.SV

            xa = self.xa

            bLo = np.ravel(xa.min(0))
            bHi = np.ravel(xa.max(0))
            g1,g2 = [np.linspace(bLo[g], bHi[g], gsize).reshape(-1,1) for g in range(2)]
            gg1,gg2 = np.meshgrid(g1,g2)

            ggx   = np.column_stack((gg1.reshape(-1,1), gg2.reshape(-1,1)))
            K     = self.rho_s * self.create_kernel_matrix(SV(ggx),SV(xs), r=rho_r).sum(1).reshape(-1,gsize)

            if ax is None:
                fig = plt.figure(figsize=figsize, dpi=dpi)
                ax  = fig.add_subplot(1,1,1)
            ax.contourf(gg1,gg2, K, cmap=dawn_cmap(), levels=10, alpha=0.75)
            ax.scatter(xa[:,0],xa[:,1], marker='2', linewidth=0.75, facecolor='orangered', zorder=1, alpha=0.85)
            ax.scatter(x[1:,0], x[1:,1], marker='s', facecolor='cyan',edgecolor='blue',linewidth=1, zorder=2)
            ax.scatter(xs[0,0], xs[0,1], marker='*', facecolor='cyan',edgecolor='blue',linewidth=1, zorder=3,s=500)
            ax.axis('square')
            ax.set_title('Adjacency-Adaptive Covariance (at Starred Point)')



























    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - DEEPGP
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """


    # [1] - python package import
    # ==============================
    import numpy as np
    import GPy, deepgp
    # from IPython.display import display



    # from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
    '''from algorithms import euclidean_distances#, manhattan_distances'''
    import sys,os
    # from sklearn.preprocessing import StandardScaler, MinMaxScaler

    # [2] - self-made packages
    # ============================================
    '''
    from algorithms import get_kernel_data_bounds as get_kernel_data_bounds
    from auxiliaries import grabBrackets
    '''

    class multiGP:

        def __init__(self, kernel = ['gaussian','matern'][1], name = 'gp', hypers = None):
            # import GPy,deepgp
            self.name = name
            self.kernel=kernel


        def get_normData(self, xIn, yIn):
            self.xBounds = [[xIn[:,g].min() for g in range(xIn.shape[1])],[xIn[:,g].max() for g in range(xIn.shape[1])]]
            self.yBounds = [[yIn[:,g].min() for g in range(yIn.shape[1])],[yIn[:,g].max() for g in range(yIn.shape[1])]]

        def normDown(self, zIn, mode = ['x','y'][0]):

            zscale = [self.xBounds if 'x' in mode else self.yBounds][0]
            zOut = zIn * 0
            for g in range(zOut.shape[1]):

                xg = zIn[:,g]
                xg = (xg - zscale[0][g])/(zscale[1][g]-zscale[0][g])
                zOut[:,g] = xg

            return zOut


        def normUp(self, zIn, mode = ['x','y'][0]):

            zscale = [self.xBounds if 'x' in mode else self.yBounds][0]
            zOut = zIn * 0
            for g in range(zOut.shape[1]):

                xg = zIn[:,g]
                xg = (xg)*(zscale[1][g]-zscale[0][g]) +zscale[0][g]
                zOut[:,g] = xg

            return zOut


        def fit(self, x, y, xa = None, layerDims = [], trainIters = 500, nSeeds = 0, verbose = False, xe=None, hypers=None):

            # [note] - (5)-deepgp!-[10]-{@._drr}
            # this means 5% inducing points, ARD training, 10 dimensional latent space, AAGP vectorizations.
            zIn = x.copy()
            inducers = 1
            if '(' in self.name:
                inducers = int(grabBrackets(self.name, key='('))/100



            xa = [x if xa is None else xa][0]

            self.x = x
            self.y = y
            self.xa = xa


            xIn= x.copy()
            xPaster = lambda xIn: xIn

            idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds = get_kernel_data_bounds(x, y, xa=xa)


            v  = np.square(y-y.T)
            vt = v[np.tril_indices(y.shape[0],k=-1)]
            d,da = [euclidean_distances(g,g) for g in [x,xa]]
            dt,dat = [g[np.tril_indices(g.shape[0],k=-1)] for g in [d,da]]

            rMin = raBounds[0]
            # rMin = dt[dt>0].min()
            rMax = raBounds[1]
            # sMin = np.percentile(vt[vt>0],25)
            # sMax = np.percentile(vt[vt>0],75)
            sMin = sBounds[0]
            sMax = sBounds[1]
            nMin = nBounds[0]
            nMax = nBounds[1]

            if 'deep' in self.name:
                rMin = (rMin - raBounds[0])/(raBounds[1]-raBounds[0])
                rMax = (rMax - raBounds[0])/(raBounds[1]-raBounds[0])
                sMin = (vt - sBounds[0])/(sBounds[1]-sBounds[0])
                sMin = sMin[sMin>0].min()
                sMax = (sMax - sBounds[0])/(sBounds[1]-sBounds[0])
                nMin = 0
                nMax = 1e-2

                # print(['%0.3f'%(g) for g in [rMin,rMax, sMin,sMax, nMin,nMax]])


            ARD = '!' in self.name
            vectorizer = '@' in self.name

            if vectorizer == True:

                '''
                Func = lambda xIn: xIn
                a  = adjacencyVectorization(vType = 'ard', functionOnly = 1, CFMODE=0)
                SV = a.analyze(x,y,xa=xa,f_x=Func)
                '''
                SV,SR = self.fit_surrogate(x,y,xa=xa)
                x_sv = SV(x)
                # xIn  = np.matrix(np.column_stack((x,x_sv)))
                # xPaster = lambda xIn: np.matrix(np.column_stack((xIn,SV(xIn))))

                xIn = x_sv
                xPaster = lambda xIn: np.matrix(SV(xIn))

                zIn  = xIn.copy()
                self.supReg = SR
            # if inducers < 1:
            #     zIn,_ = geoSpace(xIn.copy(),m=int(np.ceil(inducers * x.shape[0])))


            if '[' in self.name and 'deep' in self.name:

                # layerDims = [int(g) for g in self.grabBrackets(self.name,key='[').split('_')]
                layerDims = grabBrackets(self.name,key='[').split('_')
                # print(layerDims)
                ldims = []
                for h in layerDims:

                    if not '+' in h:
                        g = int(h)
                        if g == 0:
                            g = xIn.shape[1]
                        if g < 0:
                            g = int(np.ceil(xIn.shape[1]/np.abs(g)))
                    if '+' in h:
                        g = int(h) * xIn.shape[1]


                    ldims.append(g)
                layerDims = ldims


            ZPOINTS = int(np.max([2,int(inducers * x.shape[0])]))
            kBase = [GPy.kern.Matern52 if 'mat' in self.kernel else GPy.kern.RBF][0](xIn.shape[1], ARD=ARD, )# + GPy.kern.Bias(xIn.shape[1])
            if 'deep' in self.name:

                kPuts = []
                for i in range(len(layerDims)):
                    k = [GPy.kern.Matern52 if 'mat' in self.kernel else GPy.kern.RBF][0](layerDims[i], ARD=ARD, )# + GPy.kern.Bias(layerDims[i])
                    kPuts.append(k)
                kPuts.append(kBase)
                hLayers = len(kPuts)
                
                dimensions = [y.shape[1]] + layerDims + [xIn.shape[1]]

                self.get_normData(xIn,y)
                xIn = self.normDown(xIn, mode='x')
                



                # MODEL = deepgp.DeepGP(dimensions, y.A, xIn.A, kernels = kPuts, num_inducing = int(np.ceil(inducers*x.shape[0])), back_constraint = False,inits = 'None',normalize_Y=True)
                MODEL = deepgp.DeepGP(dimensions, y, xIn, kernels = kPuts, num_inducing = ZPOINTS, back_constraint = False,inits = 'None',normalize_Y=False, shuffle = False)

            else:
                # MODEL = GPy.models.SparseGPRegression(X = xIn.A, Y = y.A, kernel = kBase, num_inducing = int(np.ceil(inducers*x.shape[0])))
                MODEL = GPy.models.SparseGPRegression(X = xIn, Y = y, kernel = kBase, num_inducing =ZPOINTS )


            # [1] - constrain the bounds.

            if verbose == False:
                sys.stdout = open(os.devnull, "w")
                sys.stderr = open(os.devnull, "w")

            MODEL['.*lengthscale'].constrain_bounded(rMin,rMax)
            MODEL['.*variance'].constrain_bounded(sMin,sMax)
            # MODEL['.*variance'].constrain_fixed(1)
            MODEL['.*Gaussian_noise'].constrain_bounded(nMin,nMax)
            # MODEL['.*inducing_inputs'].constrain_fixed()
            # MODEL['.*bias'].constrain_bounded(nMin,nMax)

            # >>> m.parameter_names()
            # ['obslayer.inducing inputs', 'obslayer.Mat52.variance', 'obslayer.Mat52.lengthscale', 'obslayer.Gaussian_noise.variance', 'obslayer.Kuu_var', 'obslayer.latent space.mean', 'obslayer.latent space.variance']
            # >>> m['.*Kuu_var']
            # ←[1mdeepgp.obslayer.Kuu_var←[0;0m:
            # Param([0.00094017, 0.00054844, 0.0005889 , 0.00038871, 0.00064148,
            #        0.00055227, 0.00061474, 0.00094146])


            if verbose == False:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

            # [2] - optimize

            MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer = 'lbfgs')
            # MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer='scg')
            # MODEL.optimize(max_iters=trainIters, messages=verbose,optimizer='tnc')
            if nSeeds > 0:
                MODEL.optimize_restarts(num_restarts = nSeeds, verbose = verbose)

            if verbose == False:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            def inference_model(predictor, xIn):
                groupings = np.array_split(list(range(xIn.shape[0])), int(np.ceil(np.sqrt(xIn.shape[0]))))
                yOut      = np.zeros((xIn.shape[0],1))

                for group in groupings:
                    xi = xIn[group,:]
                    yi = predictor(xi)
                    yOut[group,:] = yi
                return np.matrix(yOut).reshape(-1,1)

            mu_ = lambda xIn: MODEL.predict(xPaster(xIn))[0].reshape(-1,1)
            sig_= lambda xIn: MODEL.predict(xPaster(xIn))[1].reshape(-1,1)

            if 'deep' in self.name:
                mu_ = lambda xIn: np.matrix(MODEL.predict(self.normDown(xPaster(xIn), mode='x'))[0]).reshape(-1,1)
                sig_= lambda xIn: np.matrix(MODEL.predict(self.normDown(xPaster(xIn), mode='x'))[1]).reshape(-1,1)
            mu = lambda xIn: inference_model(mu_, xIn)
            sig= lambda xIn: inference_model(sig_, xIn)
            self.mu = mu
            self.sig= sig
            self.MODEL=MODEL
            self.xIn = xIn
            self.zIn = zIn

        def predict(self, xp, type_ = 'response'):

            predictions = {
                            'response': self.mu,
                            'variance': self.sig,
                            }
            pred = predictions[type_]
            return pred(xp)

        # def display(self):
        #     display(self.MODEL)









    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - OPTIMIZATION
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """

    # [1] - package imports
    # =========================
    import numpy as np
    from scipy.optimize import minimize as scipopt




    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    rs = RandomState(MT19937(SeedSequence(777)))


    # [2] - self-made imports
    # =========================================
    '''
    from algorithms import lhs_sampling, FFD

    from auxiliaries import VS#,CS,VSS,CSS
    '''

    # [3] - introduce auxiliary codes
    # ================================
    import warnings
    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.simplefilter(action="ignore", category=np.VisibleDeprecationWarning)


    # [4] - Function Evaluator
    # ===================================
    def functionEvaluator(x, optFuncIn, logScale = True):

        if len(x.shape)==1:
            x = x.reshape(1,-1)
        # this function evaluates the function depending on the number of inputs.
        resp  = np.zeros((x.shape[0],1)) + np.inf
        nArgs = optFuncIn.__code__.co_argcount

        for i in range(resp.shape[0]):
            xi = x.reshape(resp.shape[0],-1)[[i],:]
            if nArgs == 1:

                try:
                    r = optFuncIn(xi)
                except:
                    try:
                        r = optFuncIn([xi[0,g] for g in range(xi.shape[1])])
                    except:
                        if xi.shape[1] == 1:
                            r = optFuncIn(float(xi))
                        else:
                            r = optFuncIn(xi.A)

            else:
                r = optFuncIn(*[xi[0,g] for g in range(xi.shape[1])])
            resp[i,:] = r
            pass
        if logScale == True:
            if resp.min()> 0:
                resp = np.log(resp)
            if resp.min()== 0:
                resp = np.log(resp + 1)
        return resp



    # [5] - Particle Swarm Optimization
    # ==================================

    class PSO:
        def __init__(
                        self,
                        optFunc,
                        optBounds,
                        nSamps=20,
                        nIters=10,
                        traductive=True,
                        addParticleCorners=False,
                        r1=[0.5,0.125,0.25][0],
                        r2=[0.5,0.125,0.25][0],
                        c1=2.05,
                        c2=2.05,
                        w=0.72984,
                        haltTol=1e-3,
                        logScale=False
                    ):
            self.c1 = c1
            self.c2 = c2
            self.r1 = r1
            self.r2 = r2
            self.__optFunc__ = optFunc
            self.w = w
            self.optBounds = optBounds
            self.optFunc = lambda xIn: functionEvaluator(
                                                        self.clipBounds(xIn),
                                                        optFunc,
                                                        logScale=logScale
                                                        )
            self.traductive = traductive
            self.nSamps = int(nSamps + (2**len(optBounds[0]))*int(addParticleCorners))
            self.nIters = nIters
            self.addParticleCorners = addParticleCorners


        def clipBounds(self, xIn):
            # xIn = xIn_.reshape(-1,len(self.optBounds[0]))
            optBounds = self.optBounds
            bLo, bHi = optBounds
            for g in range(xIn.shape[1]):
                xg = xIn[:, g]
                xg[xg < bLo[g]] = bLo[g]
                xg[xg > bHi[g]] = bHi[g]
                xIn[:, g] = xg
            # print(xIn.max(),xIn.min())
            return xIn

        def create_particles(self):
            # [INTRO] - this function will create the particles required.
            AC = self.addParticleCorners
            x  = lhs_sampling(n = self.nSamps, p=len(self.optBounds[0]), iterations=10)

            if AC:
                x2 = FFD(dims=len(self.optBounds[0]), levels=2)
                x  = VS(x2,x)
            bLo,bHi = self.optBounds
            for i in range(x.shape[1]):
                xi = x[:,i]
                xi = xi * (bHi[i]-bLo[i]) + bLo[i]
                x[:,i] = xi
            return x

        def optimize(self,):
            # [REFERENCE] - D:\Research\Research Notes\Heuristic Optimization\Particle swarm algorithm for solving systems of nonlinear equations.pdf

            # [1] - lets make the particles right now.

            x = self.create_particles()
            v = np.divide(x,np.sqrt(np.square(x).sum(1)).reshape(-1,1))
            v = v*1e-3
            w = 1
            x_tracks = np.zeros((x.shape[0],x.shape[1],self.nIters))
            y_tracks = np.ones((x.shape[0], 1, self.nIters))*np.inf
            c1 = 1
            c2 = 1

            # [2] - now we iterate.
            for i in range(self.nIters):
                y               = self.optFunc(x)
                x_tracks[:,:,i] = x
                y_tracks[:,:,i] = y

                # this will first get the best IN ALL iterations, then find the best OF ALL iterations.
                person_best_idx = np.argmin(y_tracks,axis=2)
                personal_bests  = [y_tracks[i,:,person_best_idx[i]] for i in range(x.shape[0])]
                p = np.concatenate([x_tracks[i,:,person_best_idx[i]] for i in range(x.shape[0])], axis=0)
                # p=np.matrix(p)


                # then, we use equation 2.4 to update the worst competitor.
                # person_worst_idx= np.argmax(np.ravel(personal_bests))
                # personal_worst  = y_tracks[person_worst_idx,0,person_best_idx[person_worst_idx]].reshape(-1,1)
                # pw     = x_tracks[person_worst_idx,:,person_best_idx[person_worst_idx]]
                # pw_unit= np.sqrt(np.square(pw).sum(1))
                # pw_unit[pw_unit==0] = 1
                # pw_unit= np.divide(pw,pw_unit)
                # eps    = 1e-8

                # central differencing
                '''
                dpw_dx = self.optFunc(pw+eps*pw_unit) - self.optFunc(pw-eps*pw_unit)
                boundDiff = np.matrix(self.optBounds).max(0) - np.matrix(self.optBounds).min(0)
                boundDiff[boundDiff==0] = 1
                dpw_dx = np.divide(dpw_dx, 2*eps * boundDiff)
                '''

                # forward differencing
                # dpw_dx = self.optFunc(pw+eps*pw_unit) - personal_worst
                # boundDiff = np.matrix(self.optBounds).max(0) - np.matrix(self.optBounds).min(0)
                # boundDiff[boundDiff==0] = 1
                # dpw_dx = np.divide(dpw_dx, eps * boundDiff)


                global_best_idx = np.argmin(np.ravel(personal_bests))
                global_best     = y_tracks[global_best_idx,0,person_best_idx[global_best_idx]]
                g               = x_tracks[global_best_idx,:,person_best_idx[global_best_idx]]
                if i == self.nIters-1:
                    break


                # print([g.shape for g in [p,x,g]])
                # [3] - then, we follow equation 2.1 and 2.2 from the above paper.
                # [3] - according to the authors, the PSO algorithm converges fast but slows down.

                if self.traductive:
                    w  = 0.4 * (i+1 - self.nIters) / np.square(self.nIters) + 0.4
                    c1 = -3 * (i+1)/self.nIters+3.5
                    c2 = 3 * (i+1)/self.nIters+0.5

                else:
                    pass
                    # r1 = np.matrix(np.diag([np.random.rand() for g in range(2)]))
                    # r2 = np.matrix(np.diag([np.random.rand() for g in range(2)]))

                np.random.seed(777+i)
                r1 = np.matrix(np.diag([np.random.uniform() for g in range(x.shape[0])]))
                r2 = np.matrix(np.diag([np.random.uniform() for g in range(x.shape[0])]))
                v = w * v + c1 * r1 @ (p-x) + c2 * r2 @ (g-x)
                x = x + v

            self.xOpt = g
            self.yOpt = global_best
            self.x_track = x_tracks
            self.y_track = y_tracks

            return np.matrix(g), np.matrix(global_best)
        









    def class_execute_optimizer(optFunc, optBounds, O='pso', addCorners=False, ns=150, ni=5):
        # [intro] - This function executes the optimizer in a self-contained way so we dont have to write lines of code every time jesusChrist


        OPTIMIZER = PSO(optFunc, optBounds, addParticleCorners=addCorners, nSamps = ns, nIters=ni)
        xOpt, yOpt = OPTIMIZER.optimize()
        xOpt = np.ravel(xOpt)
        return OPTIMIZER, xOpt, yOpt



    def SPOPT_EXE(optFunc, optBounds, method = ['Powell','Nelder-Mead','L-BFGS-B'][0], maxiter = None, maxfeval = None, extra_eval = True, prefer_pso = False):


        if prefer_pso:
            ns = int(10 * int(np.sqrt(len(optBounds[1])) + np.log(len(optBounds[0])+1)))
            OPTIMIZER, xOpt,yOpt = class_execute_optimizer(
                                                            None,
                                                            optFunc     = optFunc,
                                                            optBounds   = optBounds,
                                                            ns          = ns,
                                                            ni          = maxiter,
                                                            )
            return OPTIMIZER, xOpt,yOpt
        else:
            bLo = optBounds[0]
            bHi = optBounds[1]
            bounds = [[bLo[g], bHi[g]] for g in range(len(bLo))]
            x0  = np.ravel(np.array(optBounds).mean(0))
            x1  = np.ravel(np.array(optBounds).min(0))
            x2  = np.ravel(np.array(optBounds).max(0))

            options = None
            if not maxiter is None:
                options = {'maxiter':maxiter}
            if not maxfeval is None:
                options['maxfeval'] = maxfeval

            OPTIMIZER = scipopt(
                                    fun     = optFunc,
                                    x0      = np.ravel(x0),
                                    method  = method,
                                    bounds  = bounds,
                                    options = options,

                                )
            if extra_eval:
                O = [OPTIMIZER]
                for g in [x1,x2]:
                    o = scipopt(
                                        fun     = optFunc,
                                        x0      = np.ravel(g),
                                        method  = method,
                                        bounds  = bounds,
                                        options = options
                                    )
                    O.append(o)
                yhats = [g.fun for g in O]
                idx = np.argmin(yhats)
                OPTIMIZER = O[idx]

            return OPTIMIZER, OPTIMIZER.x, OPTIMIZER.fun












    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - VISUALIZATION
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from   matplotlib import cm
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib import ticker
    from matplotlib.colors import ListedColormap#,LinearSegmentedColormap
    from matplotlib import colors as mpl_colors

    def SETSTYLE(style=['bmh','default','seaborn','fivethirtyeight'][0], clear = True):
        try:
            if clear:
                plt.style.use('default')
            plt.style.use(style)
            mpl.rcParams['mathtext.fontset'] = 'cm'
            mpl.rcParams['font.family'] = 'STIXGeneral'
            plt.rcParams.update({'font.family': 'STIXGeneral'})
        except:
            import matplotlib.pyplot as plt
            import matplotlib as mpl
            if clear:
                plt.style.use('default')
            plt.style.use(style)

            mpl.rcParams['mathtext.fontset'] = 'cm'
            mpl.rcParams['font.family'] = 'STIXGeneral'
            plt.rcParams.update({'font.family': 'STIXGeneral'})

        return

    def rgb_to_hex(r, g, b):
        return '#{:02x}{:02x}{:02x}'.format(r, g, b)
    def hex_to_rgb(hex):
        return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

    def get_f38_palette(output_format = ['hex','rgb'][0]):
        F38_palette = ['#30a2da',
                    '#fc4f30',
                    '#e5ae38',
                    '#6d904f',
                    '#8b8b8b',
                    '#e630e6',
                    '#30309c',
                    '#423030',
                    '#30e67d',
                    '#7c4ee6',
                    '#a2305c',
                    '#bde686',
                    '#e68bb9',
                    '#7ee6e5',
                    '#8fe630'
                    ]
        if 'h' in output_format:
            return F38_palette
        else:
            return np.asarray([hex_to_rgb(g) for g in F38_palette])
        


    def get_global_colors(classic = False, maximize_visual_differencing=False, add_classic = False):
        '''
        # Get Global Colors
        This function grabs a bunch of colors and selects them such that they alternate between major color (regd, blue, green, yellow), with different shading.
        - If `maximize=True`, then all the colors will be selected in an order that maximizes their RGB distance.
        '''
        
        # [0] - instantiate the classic colors
        # ==========================================
        classic_colors = ['C%s'%(g) for g in range(0,32)] * int(add_classic)

        # [1] - grab the varieties of reds, blues, greens, and yellows
        # ==============================================================
        reds    = {
                    0: 'red',                1: 'orangered',
                    2: 'indianred',                3: 'darkred',
                    4: 'lightcoral',                5: 'firebrick',
                    6: 'tomato',                7: 'coral',
                    }
        blues    = {
                    0: 'blue',                1: 'navy',
                    2: 'slateblue',                3: 'mediumslateblue',
                    4: 'darkviolet',                5: 'magenta',
                    6: 'purple',                7: 'royalblue',
                    }
        greens    = {
                    0: 'green',                1: 'lime',
                    2: 'olivedrab',                3: 'springgreen',
                    4: 'greenyellow',                5: 'aquamarine',
                    6: 'palegreen',                7: 'seagreen',
                    }
        yellows    = {
                    0: 'gold',                1: 'orange',
                    2: 'yellow',                3: 'navajowhite',
                    4: 'goldenrod',                5: 'khaki',
                    6: 'peru',                7: 'yellowgreen'
                    }

        # [2] - grab the varieties of some "HARD" colors (4 groups of red, green, blue, yellow)
        # ===================================================
        hard_colors = [
            # group 1
            # ........
            'blue','gold','red','green',

            # group 2
            # ........
            'navy','peru','magenta','lime',

            # group 3
            # ........
            'deepskyblue','orange','deeppink','springgreen',

            # group 4
            # ........
            'aquamarine','yellow','sandybrown','olivedrab',
        ]

        # [3] - generate the coloring
        # ====================================================
        new_colors = []
        for i in range(len(list(reds.keys()))):
            colors      = [g[i] for g in [blues, yellows, reds, greens]]
            new_colors  += colors
        
        output_colors = new_colors + classic_colors if not classic else classic_colors + new_colors
        output_colors = hard_colors + classic_colors + new_colors if not classic else classic_colors + hard_colors + new_colors

        # [4] - if we want to maximize the visual differences between them, then we will apply maximin distancing.
        # =============================================================================================================
        if maximize_visual_differencing:
            from sklearn.metrics.pairwise import euclidean_distances

            rgb_codes = np.array([mpl_colors.to_rgb(g) for g in output_colors])*255
            x         = rgb_codes[0,:].reshape(1,-1)
            rgb_codes = np.delete(rgb_codes,0,axis=0)
            for i in range(rgb_codes.shape[0]):
                d = euclidean_distances(x,rgb_codes).min(0)
                idx = np.argmax(np.ravel(d))
                x   = np.vstack((x,rgb_codes[idx,:].reshape(1,-1)))
                rgb_codes = np.delete(rgb_codes,idx,axis=0)

            x = x.astype(int)
            output_colors = [rgb_to_hex(*x[i,:]) for i in range(x.shape[0])]
        return output_colors
            


    def dawn_cmap(reverse = False):
        '''
        This function will return a custom colormap.
        '''
        # [1] - make the coloring
        # ==============================
        mats = np.matrix('[255 255 195;255 255 194;255 255 193;255 255 191;255 255 190;255 255 189;255 255 188;255 255 187;255 255 186;255 255 185;255 255 184;255 255 183;255 255 182;255 255 181;255 255 179;255 255 178;255 255 177;255 255 176;255 255 175;255 255 174;255 255 173;255 255 172;255 255 171;255 255 170;255 255 169;255 255 167;255 255 166;255 255 165;255 255 164;255 255 163;255 255 162;255 255 161;255 255 160;255 255 159;255 255 158;255 255 157;255 255 155;255 255 154;255 255 153;255 255 152;255 255 151;255 255 150;255 255 149;255 255 148;255 255 147;255 255 146;255 255 145;255 255 143;255 255 142;255 255 141;255 255 140;255 255 139;255 255 138;255 255 137;255 255 136;255 255 135;255 255 134;255 255 133;255 255 131;255 255 130;255 255 129;255 255 128;255 255 127;255 255 126;255 255 125;255 253 125;255 251 125;255 249 125;255 247 125;255 245 125;255 242 125;255 241 125;255 238 125;255 237 125;255 235 125;255 233 125;255 231 125;255 229 126;255 227 126;255 225 126;255 223 126;255 221 126;255 219 126;255 217 126;255 215 126;255 213 126;255 211 126;255 209 126;255 207 126;255 205 126;255 203 126;255 201 126;255 199 126;255 197 126;255 195 126;255 193 126;255 191 126;255 189 126;255 187 126;255 185 126;255 183 126;255 181 126;255 179 126;255 177 126;255 175 126;255 173 126;255 171 126;255 169 126;255 167 126;255 165 126;255 163 126;255 161 126;255 159 126;255 157 126;255 155 126;255 153 126;255 151 126;255 149 126;255 147 126;255 145 127;255 143 127;255 141 127;255 138 127;255 136 127;255 134 127;255 132 127;255 131 127;255 129 127;254 126 127;252 125 127;250 122 127;248 121 127;246 118 127;244 116 127;242 115 127;240 113 127;238 111 127;236 109 127;234 107 127;232 105 127;230 102 127;228 100 127;226 98 127;224 97 127;222 94 127;220 93 127;218 91 127;216 89 127;214 87 127;212 84 127;210 83 127;208 81 127;206 79 127;204 77 127;202 75 127;200 73 127;198 70 127;196 68 127;194 66 127;192 64 127;190 63 127;188 61 127;186 59 127;184 57 127;182 54 127;180 52 127;178 51 127;176 49 127;174 47 127;171 44 127;169 42 127;167 40 127;165 39 127;163 37 127;161 34 127;159 33 127;157 31 127;155 29 127;153 27 127;151 25 127;149 22 127;147 20 127;145 18 127;143 17 127;141 14 127;139 13 127;137 11 127;135 9 127;133 6 127;131 4 127;129 2 127;127 0 127;125 0 127;123 0 127;121 0 127;119 0 127;117 0 127;115 0 127;113 0 127;111 0 127;109 0 127;107 0 127;105 0 127;103 0 127;101 0 127;99 0 127;97 0 127;95 0 127;93 0 127;91 0 127;89 0 127;87 0 126;85 0 126;83 0 126;82 0 126;80 0 126;78 0 126;76 0 126;74 0 126;72 0 126;70 0 126;68 0 126;66 0 126;64 0 126;62 0 126;60 0 126;58 0 126;56 0 126;54 0 126;52 0 126;50 0 126;48 0 126;46 0 126;44 0 126;42 0 126;40 0 126;38 0 126;36 0 126;34 0 126;32 0 126;30 0 126;28 0 126;26 0 126;24 0 126;22 0 126;20 0 126;18 0 126;16 0 126;14 0 126;12 0 126;10 0 126;8 0 126;6 0 126;4 0 126;2 0 126;0 0 126]')
        mats = np.flipud(mats)

        # [2] - scale between 0-1
        # ==============================
        mats = mats/mats.max()
        if reverse:
            mats = np.flipud(mats)
        
        # [3] - generate the palette
        # ==============================
        cmap = ListedColormap(mats)

        # [4] - return
        # ==============================
        return cmap

    def pretty_plot(ax, fig, gg1, gg2, gg3, projection=['2d','3d'], cmap=cm.jet, fill_2d_contour = True, add_2d_colorbar=False,):
        levSize1 = 100
        levSize2 = 10
        lev      = lambda g, levs: np.linspace(g.min(), g.max(), levs).tolist()


        if projection.lower() == '3d':
            alpha = 0.75
            ax.plot_surface(gg1,gg2,gg3,
                            rstride     = 1,
                            cstride     = 1,
                            cmap        = cmap,
                            alpha       = alpha,
                            antialiased = True,
                            shade       = True,
                            edgecolor   = 'none'
                            )
            ax.plot_wireframe(gg1,gg2,gg3,
                            rstride       = 1,
                            cstride       = 2,
                            linewidth     = 0.25,
                            color         = 'black',
                            alpha         = 0.125,
                            antialiased   = True
                            )
            try:
                ax.contour(gg1,gg2,gg3, cmap = cmap, alpha = 0.5, offset = gg3.min())
            except:
                pass
        else:
            if fill_2d_contour:
                # try:
                    surface = ax.contourf(gg1,gg2,gg3,
                                        lev(gg3, levSize1),
                                        cmap = cmap,
                                        alpha= 0.75
                                        )
                    ax.contour(gg1,gg2,gg3,
                            lev(gg3,levSize2),
                            cmap = cmap,
                            alpha= 1.0,
                            )
                # except:
                #     print('WARNING. No contour levels.')
            else:
                surface = ax.contourf(gg1,gg2,gg3,
                                    lev(gg3, levSize1),
                                    cmap = cmap,
                                    alpha = 0.375
                                    )
                ax.contour(gg1,gg2,gg3,
                        lev(gg3, levSize2), 
                        cmap = cmap,
                        linewidths = 1,
                        zorder = -1,
                        alpha = 1.0
                        )
            if add_2d_colorbar:
                divider = make_axes_locatable(ax)
                cax     = divider.append_axes('right',size = '5%', pad = 0.05)
                cb = fig.colorbar(surface, cax = cax, orientation = 'vertical')
                tick_locator = ticker.MaxNLocator(nbins=4)
                cb.locator   = tick_locator
                cb.update_ticks()
        return ax,fig



    def fast_plot(ax, fig, bounds, func, projection=['2d','3d'][1],cmap = cm.jet, gsize=50, view_elevation = 15, view_rotation=330, square_axes=True, add_2d_colorbar=False):
        if type(bounds) != list:
            bounds = [bounds.min(0).tolist(), bounds.max(0).tolist()]
        
        # [1] - get the bounds and make the grid
        # ==========================================
        bLo,bHi = bounds
        g1,g2 = [np.linspace(bLo[g], bHi[g], gsize) for g in range(len(bLo))]
        gg1,gg2 = np.meshgrid(g1,g2)
        ggx     = np.vstack((np.ravel(gg1), np.ravel(gg2))).T
        ggy     = func(ggx)
        gg3     = np.asarray(ggy.reshape(-1,gsize))
        # [2] - plot the mesh    
        # ==========================================
        ax,fig = pretty_plot(ax, fig, gg1,gg2,gg3, projection = projection, cmap = cmap,add_2d_colorbar=add_2d_colorbar)
        if projection.lower() == '3d':
            ax.view_init(view_elevation, view_rotation)
        else:
            # plt.tight_layout()
            if square_axes:
                ax.set_aspect(1 / ax.get_data_ratio())
        
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlabel(r'$X_{1}$')
        ax.set_ylabel(r'$X_{2}$')
        try:
            ax.set_zlabel(r'$Y$')
            ax.set_zticklabels([])
        except:
            pass
        return ax,fig




    def dawn_cmap(reverse = False):

        # reads in duskmap.txt and returns as a cmap.
        from matplotlib.colors import ListedColormap,LinearSegmentedColormap



        mats = np.matrix('[255 255 195;255 255 194;255 255 193;255 255 191;255 255 190;255 255 189;255 255 188;255 255 187;255 255 186;255 255 185;255 255 184;255 255 183;255 255 182;255 255 181;255 255 179;255 255 178;255 255 177;255 255 176;255 255 175;255 255 174;255 255 173;255 255 172;255 255 171;255 255 170;255 255 169;255 255 167;255 255 166;255 255 165;255 255 164;255 255 163;255 255 162;255 255 161;255 255 160;255 255 159;255 255 158;255 255 157;255 255 155;255 255 154;255 255 153;255 255 152;255 255 151;255 255 150;255 255 149;255 255 148;255 255 147;255 255 146;255 255 145;255 255 143;255 255 142;255 255 141;255 255 140;255 255 139;255 255 138;255 255 137;255 255 136;255 255 135;255 255 134;255 255 133;255 255 131;255 255 130;255 255 129;255 255 128;255 255 127;255 255 126;255 255 125;255 253 125;255 251 125;255 249 125;255 247 125;255 245 125;255 242 125;255 241 125;255 238 125;255 237 125;255 235 125;255 233 125;255 231 125;255 229 126;255 227 126;255 225 126;255 223 126;255 221 126;255 219 126;255 217 126;255 215 126;255 213 126;255 211 126;255 209 126;255 207 126;255 205 126;255 203 126;255 201 126;255 199 126;255 197 126;255 195 126;255 193 126;255 191 126;255 189 126;255 187 126;255 185 126;255 183 126;255 181 126;255 179 126;255 177 126;255 175 126;255 173 126;255 171 126;255 169 126;255 167 126;255 165 126;255 163 126;255 161 126;255 159 126;255 157 126;255 155 126;255 153 126;255 151 126;255 149 126;255 147 126;255 145 127;255 143 127;255 141 127;255 138 127;255 136 127;255 134 127;255 132 127;255 131 127;255 129 127;254 126 127;252 125 127;250 122 127;248 121 127;246 118 127;244 116 127;242 115 127;240 113 127;238 111 127;236 109 127;234 107 127;232 105 127;230 102 127;228 100 127;226 98 127;224 97 127;222 94 127;220 93 127;218 91 127;216 89 127;214 87 127;212 84 127;210 83 127;208 81 127;206 79 127;204 77 127;202 75 127;200 73 127;198 70 127;196 68 127;194 66 127;192 64 127;190 63 127;188 61 127;186 59 127;184 57 127;182 54 127;180 52 127;178 51 127;176 49 127;174 47 127;171 44 127;169 42 127;167 40 127;165 39 127;163 37 127;161 34 127;159 33 127;157 31 127;155 29 127;153 27 127;151 25 127;149 22 127;147 20 127;145 18 127;143 17 127;141 14 127;139 13 127;137 11 127;135 9 127;133 6 127;131 4 127;129 2 127;127 0 127;125 0 127;123 0 127;121 0 127;119 0 127;117 0 127;115 0 127;113 0 127;111 0 127;109 0 127;107 0 127;105 0 127;103 0 127;101 0 127;99 0 127;97 0 127;95 0 127;93 0 127;91 0 127;89 0 127;87 0 126;85 0 126;83 0 126;82 0 126;80 0 126;78 0 126;76 0 126;74 0 126;72 0 126;70 0 126;68 0 126;66 0 126;64 0 126;62 0 126;60 0 126;58 0 126;56 0 126;54 0 126;52 0 126;50 0 126;48 0 126;46 0 126;44 0 126;42 0 126;40 0 126;38 0 126;36 0 126;34 0 126;32 0 126;30 0 126;28 0 126;26 0 126;24 0 126;22 0 126;20 0 126;18 0 126;16 0 126;14 0 126;12 0 126;10 0 126;8 0 126;6 0 126;4 0 126;2 0 126;0 0 126]')
        mats = np.flipud(mats)
        mats = mats/mats.max()
        if reverse:
            mats = np.flipud(mats)
        cmap = ListedColormap(mats)

        return cmap














































    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - MODELING
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """
    # [1] - package import
    # ============================
    import numpy as np
    from sklearn.linear_model import ARDRegression, BayesianRidge
    import warnings
    from sklearn.exceptions import ConvergenceWarning, DataConversionWarning, UndefinedMetricWarning
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    import xgboost
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, f1_score

    # [2] - self-made packages
    # ============================
    '''
    from kernel_training import kernelModel
    from algorithms import (
        CL,
        euclidean_distances,
        # trainTest_binSplitter,
        GBAL_acq,
        GREEDY_acq,
        # mutual_euclid,
        subsampler,
        SLV,
        VS,
        myKernel,
        trainTest_binSplitter,
        maxiMin_acquisition
    )

    from deepGP import (
        multiGP
    )

    from optimization import (
        class_execute_optimizer as class_execute_optimizer,
        SPOPT_EXE
    )

    from SETTINGS import (
        deepGP_maxIters
    )
    '''
    KERNEL = 'matern'










    ['r2', 'mae', 'mse', 'rmse', 'nrmse','mape', 'mmape', 'wmape', 'smape', 'bias', 'adjusted_r2', 'f1', 'cod', 'mbd', 'cv']
    def calculate_regression_metrics(y_true, y_pred, p, metric='r2'):
        """
        Calculate various regression performance metrics based on the input parameter 'metric'.

        Parameters:
        y_true (array-like): True (actual) values.
        y_pred (array-like): Predicted values.
        metric (str): The metric to calculate. Options: 'r2', 'mae', 'mse', 'rmse', 'mape', 'mmape', 'wmape', 'bias', 'adjusted_r2',
                    'f1', 'cod', 'mbd', 'cv'.

        Returns:
        float: The calculated regression performance metric.
        """
        if metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'mae':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'mse':
            return mean_squared_error(y_true, y_pred)
        elif metric == 'rmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'nrmse':
            return np.sqrt(mean_squared_error(y_true, y_pred))/(y_true.max()-y_true.min())
        elif metric == 'mape':
            absolute_percentage_errors = np.abs((y_true - y_pred) / np.abs(y_true))
            return np.mean(absolute_percentage_errors)
        elif metric == 'mmape':
            eps = 1e-10
            absolute_percentage_errors = np.abs((y_true - y_pred) / np.abs(y_true) + eps)
            return np.mean(absolute_percentage_errors)
        elif metric == 'wmape':
            wmape = np.abs((y_true - y_pred)).sum() / np.abs(y_true).sum()
            return wmape
        
        elif metric == 'smape':
            yp = np.ravel(y_pred)
            ye = np.ravel(y_true)
            error = ye - yp
            smape   = np.divide(
                                np.ravel(np.abs(error)),
                                (np.abs(yp) + np.abs(ye))/2
                                ).mean()
            return smape
            
        elif metric == 'bias':
            error   = y_true - y_pred
            overEst = np.where(error>0, np.abs(error),0)
            underEst= np.where(error<=0, np.abs(error),0)
            return np.divide(overEst-underEst,overEst+underEst).mean()

        elif metric == 'adjusted_r2':
            n = len(y_true)
            r2 = r2_score(y_true, y_pred)
            adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
            return adjusted_r2
        elif metric == 'f1':
            # Placeholder values for binary classification (not a typical regression metric)
            y_true_binary = (y_true > 0).astype(int)
            y_pred_binary = (y_pred > 0).astype(int)
            return f1_score(y_true_binary, y_pred_binary)
        elif metric == 'cod':
            variance_residuals = np.var(y_true - y_pred)
            variance_y = np.var(y_true)
            return 1 - (variance_residuals / variance_y)
        elif metric == 'mbd':
            return np.mean(y_true - y_pred)
        elif metric == 'cv':
            std_residuals = np.std(y_true - y_pred)
            mean_y = np.mean(y_true)
            return (std_residuals / mean_y) * 100

        else:
            raise ValueError("[ %s CHOSEN ] - Invalid metric. Choose from 'r2', 'mae', 'mse', 'rmse', 'mape', 'mmape', 'wmape', 'bias', 'adjusted_r2', 'f1', 'cod', 'mbd', 'cv'."%(metric))









    def fit_gp(x,y,xa=None,kernel = KERNEL, model_name = ['gp','slrgp','aagp','lrk'][0],acquisition=[None,'gbal','mm'][0]):

        

        def ALC_MSE_acquisition(sig, x, xaIn, xIn, do_fast = True, acq = ['alc','mmse','imse'][0]):
            # we will do ALC assuming xaIn is < xa >
            ranger = range(xIn.shape[0])
            scores = []
            s  = float(CKM(x[[0],:],x[[0],:]))

            ALC = 'alc' in acq
            IMSE= 'imse' in acq
            MMSE= 'mmse' in acq

            scores = []
            D      = x
            X      = xa
            Z      = xIn
            if do_fast==True:
                Kxd = CKM(X,D)
                Kxx = CKM(X,X)
                Kxz = CKM(X,Z)

                H0  = Kxd @ Kxd.T + n @ Kxx
                H0i = SLV(H0)
                v0  = np.diag(Kxx @ H0i @ Kxx.T).reshape(-1,1)
                for i in range(Z.shape[0]):
                    idx = np.where(np.ravel((Z[i,:]-X).sum(1)==0))[0]
                    jdx = [g for g in range(X.shape[0]) if not g in idx]
                    # using sherman-woodbury matrix update.
                    # ADD: [H + q.T * q]^-1 = Hi - [Hi*q.T*q*HI]/[1 + q*HI*q.T]
                    # DEL: [H - q.T * q]^-1 = Hi + [Hi*q.T*q*HI]/[1 - q*HI*q.T]
                    q = Kxz[:,i]
                    H1i = H0i + H0i @ q @ q.T @ H0i/float(s- q.T @ H0i @ q)
                    v1  = np.diag(Kxx * H1i * Kxx.T).reshape(-1,1)

                    vRed= (v0-v1)[jdx,:].sum()
                    scores.append(vRed)

            else:
                I2 = np.matrix(np.eye(D.shape[0]+1))
                v0 = sig(X)
                Kpp= CKM(X,X)
                for i in range(Z.shape[0]):
                    Zi    = Z[i,:]
                    idx   = np.where(np.ravel((Zi-X).sum(1)) == 0)[0]
                    jdx = [g for g in range(X.shape[0]) if not g in idx]

                    D2 = VS(D,Zi)
                    H2 = CKM(D2,D2) + n * I2
                    Kps= CKM(X,D2)

                    v2 = np.diag(Kpp - Kps @ SLV(H2,Kps.T)+n).reshape(-1,1)

                    if ALC or IMSE:
                        vRed = (v0[jdx,:]-v2[jdx,:]).sum()
                    else:
                        vRed = v0[jdx,:].max()-v2[jdx,:].max()
                    scores.append(vRed)

            return np.matrix(scores).reshape(-1,1)


        def MI_acquisition(sig, x, xaIn, xIn, do_fast=True,):

            # this is mutual information.
            s = float(CKM(x[[0],:],x[[0],:]))
            scores = []
            if do_fast==True:
                Kxs = CKM(xaIn,x)
                Kxr = CKM(xaIn,xaIn)
                Kxp = CKM(xaIn, xIn)
                Kxx = CKM(xaIn,xaIn)

                H_s = Kxs @ Kxs.T + n * Kxx
                H_r = Kxr @ Kxr.T + n * Kxx

                H_si= SLV(H_s)
                H_ri= SLV(H_r)

                for i in range(xIn.shape[0]):
                    H_ri2 = np.matrix(np.copy(H_ri))
                    Kxp_i = Kxp[:,i]
                    idx   = np.where(np.ravel((xIn[i,:]-xaIn).sum(1)) == 0)[0]
                    if len(idx) > 0:

                        # using sherman-woodbury matrix update.
                        # ADD: [H + q.T * q]^-1 = Hi - [Hi*q.T*q*HI]/[1 + q*HI*q.T]
                        # DEL: [H - q.T * q]^-1 = Hi + [Hi*q.T*q*HI]/[1 - q*HI*q.T]

                        for j in idx:
                            Kxp_j = Kxx[:,j]
                            H_ri2 = H_ri + (H_ri2 @ Kxp_j @ Kxp_j.T @ H_ri2)/(s + Kxp_j.T @ H_ri2 @ Kxp_j)

                    minf = float(Kxp_i.T @ H_si @ Kxp_i)/float(Kxp_i.T @ H_ri2 @ Kxp_i)
                    scores.append(minf)
            return np.matrix(scores).reshape(-1,1)


        def train_SLRGP(
                                x,
                                y,
                                xa,
                                mu0,
                                sig0,
                                H0,
                                HI0,
                                ckm,
                                clip_variance = False
                                ):
            # [1] - predict the outputs at ALL locations
            ya = mu0(xa)

            # [2] - calculate the Laplacian
            Ly = CL(np.square(euclidean_distances(ya,ya)), normalize=False)
            Lx = CL(np.square(euclidean_distances(xa,xa)), normalize=False)
            L  = np.divide(Ly,Lx) * n
            L[~np.isfinite(L)] = 0

            # [3] - iterate through the points.
            v0 = sig0(xa)
            if clip_variance:
                v0 = np.clip(v0,0,None)
            Kxs  = ckm(xa,x)
            Kxx  = ckm(xa,xa)
            reds = []

            lambdas = [10**g for g in [-6,-5,-4,-2,-2,-1,0,1,2,3]]
            for i in range(len(lambdas)):

                l  = lambdas[i]
                H1 = H0 + l * HI0 @ Kxs.T @ L @ Kxs

                v1 = np.diag(Kxx - Kxs @ SLV(H1, Kxs.T)).reshape(-1,1)
                idx= np.argmax(np.ravel(v1))

                x2 = xa[idx,:].reshape(1,-1)
                if i == 0:
                    xstatic = x2
                if i > 0 and np.abs(x2-xstatic).sum() == 0:
                    continue

                x2 = np.vstack((x,x2))
                K2 = ckm(x2,x2)
                Kx2= ckm(xa,x2)

                H2 = K2 + np.eye(K2.shape[0])*n
                v2 = np.diag(Kxx - Kx2 @ SLV(H2, Kx2.T)).reshape(-1,1) + n

                if clip_variance:
                    v2 = np.clip(v2,0,None)

                v2 = np.delete(v2,idx,axis=0)
                v3 = np.delete(v0,idx,axis=0)
                vred = (v3-v2).mean()
                reds.append(float(vred))
            idx = np.ravel(np.argmax(reds))[0]

            lOpt = lambdas[idx]
            return L * lOpt

        def SLRGP_acquisition(
                                x,
                                y,
                                xa,
                                ckm,
                                L,
                                H0,
                                HI0,
                                xIn,
                                l=1e-9,
                                clip_variance = True
                                ):

            # [1] - create the first hessian
            Kxs = ckm(xa,x)
            Kps = ckm(xIn,x)
            Kpp = ckm(xIn,xIn)
            H1  = H0 + HI0 @ Kxs.T @ L @ Kxs

            # [2] - calculate the variance over the set of points.
            v1 = np.diag(Kpp - Kps @ SLV(H1, Kps.T)).reshape(-1,1) + n

            # [3] - iterate through all points.
            scores = xIn[:,0].reshape(-1,1)*0
            for i in range(xIn.shape[0]):
                x2 = np.vstack((x,xIn[i,:].reshape(1,-1)))
                Kss2 = ckm(x2,x2)
                Kps2 = ckm(xIn,x2)
                Kxs2 = ckm(xa,x2)
                H2  = Kss2 + np.eye(Kss2.shape[0]) * n
                H2  = H2 + SLV(H2, Kxs2.T @ L @ Kxs2)

                v2 = np.diag(Kpp - Kps2 @ SLV(H2, Kps2.T) + n).reshape(-1,1)

                grab = np.column_stack((v1,v2))
                if clip_variance:
                    grab = np.clip(grab,0,None)

                red = np.delete(grab, i, axis=0)
                red = np.mean(-red[:,1] + red[:,0])
                scores[i,:] = float(red)
            return scores



        # [1] - Train the GP
        # =======================
        tmode = 'gp'
        if 'aagp' in model_name:
            tmode = 'aagp'
        if 'lrk' in model_name:
            tmode = 'lrk'
        K = kernelModel(
            x = x,
            y = y,
            xa= xa,
            training_mode = tmode,
            kernel = kernel
        )
        CKM = K.fit()
        n   = K.nOpt


        # [2] - fit the model
        # ========================
        K = CKM(x,x)
        H = K + n * np.matrix(np.eye(x.shape[0]))
        HI= SLV(H)
        B = HI * y
        mu= lambda xIn: CKM(xIn,x) @ B
        sig=lambda xIn: np.matrix(np.diag(CKM(xIn,xIn) - CKM(xIn,x) @ HI @ CKM(x,xIn))).reshape(-1,1) + n


        # [3] - get the acquisition
        # ============================
        acq = acquisition
        if not acq is None:
            if True in [g in acq for g in ['alc','mmsee','imse']] and not 'slrgp' in acq:
                sig0 = sig
                sig=None
                sig  = lambda xIn: ALC_MSE_acquisition(sig0, x, xa, xIn, do_fast=False, acq=acq)

            if True in [g in acq for g in ['minf','minfo','mutual']]:
                sig0 = sig
                sig = None
                sig = lambda xIn: MI_acquisition(sig, x, xa, xIn, do_fast=True)

            if True in [g in acq for g in ['gbal']]:
                sig = lambda xIn: GBAL_acq(x, xIn)

            if True in [g in acq for g in ['igs','gsy']]:
                sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)
            if True in [g in model_name for g in ['slrgp']] or 'alc' in model_name or 'alc' in acquisition:
                '''[ NOTE ] - L already has the parameters applied to it!'''
                clipper = True
                L = train_SLRGP(x,y,xa,mu,sig,H,HI, CKM, clip_variance = clipper)
                sig = lambda xIn: SLRGP_acquisition(x,y,xa,CKM,L,H,HI,xIn, l=1, clip_variance=clipper)

        return mu,sig

    def fit_lod(x,y,xa=None,kernel = KERNEL, model_name = ['lod'][0],acquisition=[None,'gbal','mm'][0]):

        # [1] - Train the GP
        # =======================
        K = kernelModel(
            x = x,
            y = y,
            xa= xa,
            training_mode = 'lod',
            kernel = kernel
        )
        CKM,mu,sig = K.fit()

        # [2] - augment the acquisition if applicable
        # ============================================
        acq = acquisition
        if not acq is None:
            if True in [g in acq for g in ['gbal']]:
                sig = lambda xIn: GBAL_acq(x, xIn)

            if True in [g in acq for g in ['igs','gsy']]:
                sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)

        return mu,sig
        

    def fit_loess_regressor(x,y, train=True):

        # [INTRO] - this will fit the LOESS regression model

        def CRV(xIn, order=1):
            return PolynomialFeatures(order).fit_transform(xIn)

        def predictor(D,x, y, xIn, r=1,n=1e-3):
            K = myKernel(x,xIn, r=r, s=1, ARD=False, kernel='gaussian')
            I = np.eye(D.shape[1])
            yOut = (xIn.sum(1)*0).reshape(-1,1)
            DIn = CRV(xIn)
            for i in range(xIn.shape[0]):
                J = np.diag(np.ravel(K[:,i]))
                H = D.T @ J @ D + n * I
                HI= SLV(H)
                B = HI @ D.T @ J @ y
                yOut[i,:] = DIn[i,:] @ B
            return yOut

        def train_loess(D,x,y, hypersIn, splits):
            hin = np.ravel(hypersIn).tolist().copy()
            r=hin[0]
            n=hin[1]
            mse = []
            for i in range(len(splits)):
                rdx = splits[i]
                tdx = [g for g in range(x.shape[0]) if not g in rdx]
                Dt  = D[tdx,:]
                xt  = x[tdx,:]
                yt  = y[tdx,:]
                xr  = x[rdx,:]
                yr  = y[rdx,:]
                yp  = predictor(Dt,xt,yt, xr, r=r, n=n)
                error = np.square(yp-yr).mean()**0.5
                mse.append(error)
            return np.mean(mse)
        d           = euclidean_distances(x)
        dt          = d[d>0]
        D           = CRV(x)
        # splits      = trainTest_binSplitter(range(x.shape[0]), n=-1, x=x)
        
        ''' Good results, takes long time '''
        # splits      = np.array_split(range(x.shape[0]), int(x.shape[0]/2))

        ''' Is faster, results are ???? '''

        # original settings (loess was not using these anyway because it wasing being trained)
        # splits      = np.array_split(range(x.shape[0]), 4)
        # optBounds   = [[dt.min(),0],[dt.max()*3, 10]]

        # techno_loess_again
        splits      = np.array_split(range(x.shape[0]), 2)
        optBounds   = [[np.percentile(dt,10),1e-9],[dt.max()*3, 10]]

        # techno_loess_again2
        splits      = np.array_split(range(x.shape[0]), 2)
        optBounds   = [[dt[dt>0].min(),1e-9],[dt.max()*3, 10]]


        optProblem  = lambda hypersIn: train_loess(D,x,y, hypersIn, splits)
        xOpt        = [np.percentile(dt,5), 1e-3]
        OPTIMIZER   = None
        if train==True:
            OPTIMIZER, xOpt, yOpt = class_execute_optimizer(optFunc=optProblem, optBounds=optBounds, O='pso',ns=10, ni=3)
        rOpt,nOpt = np.ravel(xOpt).tolist()
        mu        = lambda xIn: predictor(D,x,y,xIn, r=rOpt,n=nOpt)
        return mu


    def fit_localRidge(x,y, xa=None, train=True, use_cv=True, maxiter = 4, model_name='loess', acquisition=[None,'gbal','mm'][0]):


        M = AUTOLOESS2(x,y,train=not '*' in model_name, ard_regression='ard' in model_name)
        try:
            OPTIMIZER = M.OPTIMIZER
        except:
            OPTIMIZER = None
        mu = M.mu
        sig = M.sig

        
        acq = acquisition
        if not acq is None:
            if True in [g in acq for g in ['gbal']]:
                sig = lambda xIn: GBAL_acq(x, xIn)

            if True in [g in acq for g in ['igs','gsy']]:
                sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)

        return mu,sig



    class AUTOLOESS2:
        def __init__(
                self,
                x,
                y,
                n_neighbors = -1,
                verbose=False,
                train = False,
                ard_regression = False,
        ):
            self.x = x
            self.y = y
            if n_neighbors is None:
                n_neighbors = int(np.sqrt(x.shape[0])*np.abs(np.log(np.log(x.shape[0]))))
            else:
                if n_neighbors < 2:
                    n_neighbors = int(np.sqrt(x.shape[0])*np.abs(np.log(np.log(x.shape[0]))))
            self.ard_regression = ard_regression
            self.n_neighbors = n_neighbors
            self.verbose=verbose
            self.mu = lambda xIn: self.predict(xIn)[:,0].reshape(-1,1)
            self.sig = lambda xIn: self.predict(xIn)[:,1].reshape(-1,1)
            if train:
                self.train()

        def train(self):
            def cv(x,y,ranger,idx,n_neighbors):
                
                mse = []
                for rdx in idx:
                    tdx = [g for g in ranger if not g in rdx]

                    xr  = x[rdx,:].reshape(-1,x.shape[1])
                    yr  = y[rdx,:].reshape(-1,1)

                    xt  = x[tdx,:].reshape(-1,x.shape[1])
                    yt  = y[tdx,:].reshape(-1,1)

                    # get he predictions of the data held-out.
                    yp  = self.predict(xr,x=xt,y=yt, n_neighbors=n_neighbors)[:,0]
                    err = np.square(np.ravel(yr) - np.ravel(yp)).mean()
                    mse.append(err)
                return np.mean(mse)
            


            # this function will train the model using several holdout sets.
            x = self.x
            y = self.y
            ranger = range(x.shape[0])
            splits = np.array_split(ranger, int(x.shape[0]))

            neighbor_groups = [1] + [2,] + [int(g) for g in [ np.log(x.shape[0]), np.sqrt(x.shape[0]), np.log(x.shape[0])*np.sqrt(x.shape[0]), x.shape[0] ]]
            # neighbor_groups = [2,] + [int(g) for g in [x.shape[0] * f for f in [0.25,0.5,0.75,0.95]]]
            neighbor_groups = [g for g in neighbor_groups if g >= 1]
            scores = [cv(x,y, ranger, splits, g) for g in neighbor_groups]
            nopt = neighbor_groups[np.argmin(scores)]


            self.mu = lambda xIn: self.predict(xIn, n_neighbors=nopt)[:,0].reshape(-1,1)
            self.sig = lambda xIn: self.predict(xIn, n_neighbors=nopt)[:,1].reshape(-1,1)



        def predict(self,xp, x=None,y=None, n_neighbors = None):

            if x is None or y is None:
                x = self.x
                y = self.y
            if n_neighbors is None:
                n_neighbors = self.n_neighbors
                

            # create the distance matrix
            d = euclidean_distances(xp,x)

            # create the output vector
            yout = np.zeros((xp.shape[0],2))

            # iterate
            # for i in TQDM(range(xp.shape[0]),disable = not self.verbose, desc='Fitting Local Bayesian Ridge'):
            for i in range(xp.shape[0]):
                # grab the nearest neighbors
                idx = np.argsort(np.ravel(d[i,:]))[0:n_neighbors]

                xn = x[idx,:]
                yn = np.ravel(y[idx,:])
                MM = StandardScaler()
                MM.fit(xn)
                xn = MM.transform(xn)
                # fit the model
                if self.ard_regression:
                    # M = ARDRegression(fit_intercept=True, n_iter=300)
                    try:
                        M = ARDRegression(fit_intercept=True, n_iter=300, lambda_2=100)
                    except:
                        M = ARDRegression(fit_intercept=True, max_iter=300, lambda_2=100)
                else:
                    # M = BayesianRidge(fit_intercept=True, n_iter=300)
                    try:
                        M = BayesianRidge(fit_intercept=True, n_iter=300, lambda_2=100)
                    except:
                        M = BayesianRidge(fit_intercept=True, max_iter=300, lambda_2=100)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    warnings.filterwarnings("ignore", category=DataConversionWarning)
                    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
                    warnings.filterwarnings("ignore", category=UserWarning)

                    try:
                        M.fit(xn,yn.reshape(-1,1))
                    except:
                        M.fit(xn,yn)

                # get the response surface estimate
                xi = MM.transform(xp[i,:].reshape(-1,x.shape[1]))
                yhat = M.predict(xi)

                # get the hessian
                H = M.sigma_
                HI= SLV(H)

                # get the variance
                try:
                    vhat = float(xi @ HI @ xi.T)
                except:
                    print(M.coef_)
                    H = M.sigma_
                    H = np.ravel(np.ravel(H).tolist() + [H[-1,-1]]*(xi.shape[1]-H.shape[1])).reshape(1,-1)
                    H = H.T @ H
                    HI= SLV(H)
                    vhat = float(xi @ HI @ xi.T)


                # cast
                yout[i,0] = float(yhat)
                yout[i,1] = float(vhat)
            # yout[:,1] = v0 - yout[:,1]
            return yout






    def fit_xgb_regressor(
                    x,
                    y,
                    train   = True,
                    maxiter = 50,
                    holdout = False,
                    nsplits = -1,
                    name    = 'xgb',
                    ):
        # this function will train an xgboost model and optimize the following parameters:
        # [1] - n_estimators
        # [2] - max_depth
        # [3] - colsample_bynode
        # [4] - colsample_bytree
        # [5] - colsample_bylevel
        # [6] - lambda
        # [7] - alpha
        # as we can see it is a very high dimensional feature space.. good luck!

        def trainer(x, y, ranger, idx, hypers,return_model = False):
            hin = np.ravel(hypers).tolist()
            # n_est, maxdepth, colnode, coltree, collevel, lam, alph = hin
            # colnode,coltree,collevel = [np.clip(g,0,1) for g in [colnode,coltree,collevel]]
            # n_est       = np.max([1,n_est])
            # maxdepth    = np.max([1,maxdepth])
            # lam         = np.max([0,lam])
            # alph        = np.max([0,alph])

            # n_est, maxdepth,alph = hin
            n_est,alph = hin
            maxdepth    = 15
            n_est       = np.max([1,n_est])
            maxdepth    = np.max([1,maxdepth])
            alph        = np.max([0,alph])

            # [1] - call the regressor
            m = xgboost.XGBRegressor(
                            # [ optimization inputs ]
                            n_estimators        = int(n_est),
                            max_depth           = int(maxdepth),
                            # colsample_bynode    = colnode,
                            # colsample_bytree    = coltree,
                            # colsample_bylevel   = collevel,
                            # reg_lambda          = lam,
                            reg_alpha           = alph,

                            # [ additional inputs ]
                            objective           = 'reg:squarederror',
                            # subsample           = 1,
                            # base_score          = 0.5,
                            # importance_type     = 'gain',
                            n_jobs              = 1,
                            nthread             = 1,
                            random_state        = 777,
                            seed                = 777,
                            )
            # [2] - if we want to just return the fitted model, then loets do so.
            if return_model:
                m.fit(np.asarray(x), np.asarray(y))
                mu = lambda xIn: m.predict(xIn).reshape(-1,1)
                return m,mu

            else:
                score = 0
                for i in range(len(idx)):
                    rdx = idx[i]
                    tdx = [g for g in ranger if not g in rdx]

                    xt  = x[tdx,:].reshape(-1,x.shape[1])
                    yt  = y[tdx,:].reshape(-1,1)
                    xr  = x[rdx,:].reshape(-1,x.shape[1])
                    yr  = y[rdx,:].reshape(-1,1)

                    m.fit(xt,yt)

                    err = np.square(yr - m.predict(xr)).mean() / len(idx)
                    score += err
                # print(score)
                return score


        # [1] - grab the optimization bounds
        vv  = euclidean_distances(y)
        vv  = vv[vv>0].min()
        # bLo = [2, 1, 0,0,0,0,0]
        # bHi = [200, 10, 1, 1, 1, vv, vv]
        bLo = [2, 1, 0]
        bHi = [200, 10, vv]
        bLo = [2,0]
        bHi = [200,vv]
        optBounds = [bLo,bHi]

        # [2] - create the indices for training

        ranger = list(range(x.shape[0]))
        if nsplits == -1:
            nsplits = int(np.sqrt(x.shape[0]) + np.log(x.shape[0]))
        if holdout:
            xsub= np.divide(x-x.min(0), x.max(0)-x.min(0))
            ysub= (y-y.min())/(y.max()-y.min())
            idx = [subsampler(np.column_stack((xsub,ysub)), sampling = nsplits/x.shape[0], verbose=False)[1]]
            del xsub,ysub
        else:
            idx = np.array_split(ranger, nsplits)

        # [3] - run the optimization
        optFunc = lambda hypers: trainer(x,y, ranger, idx, hypers, return_model = False)
        retFunc = lambda hypers: trainer(x,y, ranger, idx, hypers, return_model = True)
        if train:
            O, xOpt, yOpt = SPOPT_EXE(
                    optFunc = optFunc,
                    optBounds = optBounds,
                    # method = 'Nelder-Mead',
                    method = 'L-BFGS-B',
                    maxiter = maxiter,
                    extra_eval=True,
                    prefer_pso = False,
            )
            m,mu = retFunc(np.ravel(xOpt))
        else:
            nest = int(np.log(x.shape[0]) + np.sqrt(x.shape[0]))
            maxdepth = 6

            m,mu = retFunc([nest, maxdepth, 1, 1, 1, vv*1e-3, vv*1e-3])
        sig = lambda xIn: GBAL_acq(x, xIn, simple = not 'gbal' in name)
        return mu,sig




    def model_master(x,y,xa=None, model_name = 'gp',acquisition=None, kernel = None):
        if kernel is None:
            kernel = KERNEL
        if '-' in model_name:
            acquisition = model_name.split('-')[-1]


        # [1] - GP-Based models
        # ===============================================
        if True in [f in model_name for f in ['gp','slrgp','aagp','lrk']]:
            mu,sig = fit_gp(x,y,xa=xa,model_name=model_name if not 'slrgp' in model_name else 'slrgp-alc', acquisition=acquisition if not 'slrgp' in model_name else 'alc', kernel=kernel)
        
        # [2] - LOD
        # ===============================================
        elif 'lod' in model_name:
            mu,sig = fit_lod(x,y,xa=xa,model_name=model_name, acquisition=None, kernel=kernel)
        
        # [3] - LocalRidge Regression
        # ===============================================
        elif 'localridge' in model_name:
            mu,sig = fit_localRidge(x,y,xa=xa,model_name=model_name)
            if 'gbal' in model_name:
                sig = lambda xIn: GBAL_acq(x,xIn)
            else:
                sig    = lambda xIn: maxiMin_acquisition(x,xIn)


        # [4] - LOESS Regression
        # ===============================================
        elif 'loess' in model_name:
            mu = fit_loess_regressor(x,y,train=not '*' in model_name)
            if 'gbal' in model_name:
                sig = lambda xIn: GBAL_acq(x,xIn)
            else:
                sig    = lambda xIn: maxiMin_acquisition(x,xIn)
        
        # [5] - XGBoost Regression
        # ===============================================
        elif 'xgb' in model_name:
            mu,sig = fit_xgb_regressor(x,y,train=True,holdout=True,name = model_name + '-gbal')
            sig    = lambda xIn: GBAL_acq(x,xIn)
        
        
        # [6] - DeepGP
        # ===============================================
        if 'deep' in model_name and 'gp' in model_name:
            # if not '[' in model_name:
            #     model_name += ' - [10_5]'
            # M = multiGP(kernel = kernel,
            #             name = model_name,
            #             )
            reg = multiGP(name = model_name, kernel = kernel, hypers = [0.0001,0.0001],)
            reg.fit(x, y, xa = xa, trainIters = deepGP_maxIters, nSeeds = 0, verbose = False, xe=None, hypers=None, layerDims=[10,5])
            mu = reg.mu
            sig= reg.sig

        acq = acquisition
        if not acq is None:
            if True in [g in acq for g in ['gbal','maximin']]:
                sig = lambda xIn: GBAL_acq(x, xIn,simple='maximin' in acquisition)

            if True in [g in acq for g in ['igs','gsy']]:
                sig = lambda xIn: GREEDY_acq(x, mu, xIn, improveGreed = 'igs' in acq)
        return mu,sig



























    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - AUXILIARY
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """



    import numpy as np
    from joblib import Parallel
    from tqdm import tqdm as TQDM

    def VS(x1,x2):
        return np.vstack((x1,x2))
    def CS(x1,x2):
        return np.column_stack((x1,x2))
    def CSS(xList):
        return np.concatenate(xList, axis=1)
    def VSS(xList):
        return np.concatenate(xList, axis=0)

    def grabBrackets(stringIn, key = '('):

        endKey = {'(':')', '{':'}', '[':']','<':'>','/':'/'}[key]
        s1 = stringIn.index(key)+1
        s2 = stringIn.index(endKey)
        if not '/' in endKey:
            # print(stringIn[s1:s2])
            return stringIn[s1:s2]
        else:
            s2 = stringIn[s1:].index(endKey)+1
            return (stringIn[s1:])[:s2]


    class ProgressParallel(Parallel):
        def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
            self._use_tqdm = use_tqdm
            self._total = total
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            with TQDM(disable=not self._use_tqdm, total=self._total, colour='blue', bar_format="{l_bar}{bar} | {n}/{total}") as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)

        def print_progress(self):
            if self._total is None:
                self._pbar.total = self.n_dispatched_tasks
            self._pbar.n = self.n_completed_tasks
            self._pbar.refresh()





































    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - KERNEL TRAINING
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """

    # [0] - package imports
    # =====================================
    import numpy as np


    # [1] - self-made package imports
    # =====================================
    '''
    from algorithms import (
        CL,
        GET_LAPLACIAN_BOUNDS,
        get_kernel_data_bounds,
        lhs_select,
        SLOGDET,
        SLV,
        myKernel,
        # euclidean_distances,
        # trainTest_binSplitter,
    )

    from aagp import(
        AdjacencyVectorizer
    )

    from auxiliaries import (
        VS
    )

    from optimization import(
        class_execute_optimizer as class_execute_optimizer,
        # SPOPT_EXE
    )
    '''
    class kernelModel:

        def __init__(
                self,
                x,y,xa = None,
                training_mode = ['gp','aagp','lod','lrk'][0],
                kernel = ['gaussian','matern'][1],
                ARD    = False,
        ):
            self.x = x
            self.y = y
            self.xa= x.copy() if xa is None else xa
            self.training_mode = training_mode

            if ARD:
                print('Warning. Automatic-Relevance is not supported in this release. Switching to Equal-Relevance.')
                ARD = False

            self.ARD           = ARD
            self.kernel        = kernel
            self.load_data()
        
        def load_data(self):
            '''
            # Load data
            This function calculates the data driven upper and lower bounds for training.
            '''
            idxs, idxa, d, v, da, dt, vt, dat, sBounds, nBounds, rBounds, raBounds = get_kernel_data_bounds(
                x = self.x,
                y = self.y,
                xa= self.xa,
            )

            self.idxs = idxs
            self.idxa = idxa
            self.d = d
            self.v = v
            self.da = da
            self.dt = dt
            self.vt = vt
            self.dat = dat
            self.sBounds = sBounds
            self.nBounds = nBounds
            self.rBounds = rBounds
            self.raBounds = raBounds
            self.I = np.eye(self.x.shape[0])
            self.EYE = self.I.copy()
        
        def create_kernel_matrix(self, x1,x2, g=None, r=1, s=1, ARD=False):
            return myKernel(x1,x2,g=g, r=r, s=s, ARD=ARD, kernel=self.kernel)

        def fit(self):

            training_mode = self.training_mode

            if training_mode == 'gp':
                return self.train_gp()

            if training_mode == 'aagp':
                return self.train_aagp()
            
            elif training_mode == 'lod':
                return self.train_lod()
            
            elif training_mode == 'lrk':
                return self.train_lrk()
            
        
        def train_gp(self):
            '''
            # Train GP
            This function will train a kernel via the GP likelihood and return the covariance function for use.
            '''
            
            def fit_gp(x,y,hypersIn, returnLike=True):
                Hin = np.ravel(hypersIn).tolist().copy()
                s   = Hin.pop(0)
                n   = Hin.pop(0)
                r   = Hin
                if len(r) == 1:
                    r = r[0]
                ckm = lambda x1,x2: self.create_kernel_matrix(x1,x2, r=r, s=s,)

                if returnLike==False:
                    return ckm

                else:
                    K = ckm(x,x)
                    H = K + self.I * n
                    m = SLOGDET(H,y)
                    return m

            # [1] - get the bounds and data
            # ===================================
            sBounds = self.sBounds
            rBounds = self.rBounds
            nBounds = self.nBounds
            x       = self.x
            y       = self.y
            xa      = self.xa

            # [2] - set the bounds
            # ===================================
            idx = 0
            bLo = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            idx = 1
            bHi = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            optBounds = [bLo,bHi]
            
            # [3] - run the optimization
            # ====================================
            optFunc = lambda hypersIn: fit_gp(x,y,hypersIn, returnLike=True)
            retFunc = lambda hypersIn: fit_gp(x,y,hypersIn, returnLike=False)
            O,xOpt_,yOpt = class_execute_optimizer(optFunc = optFunc, optBounds=optBounds, O='pso')
            CKM  = retFunc(xOpt_)
            xOpt = np.ravel(xOpt_).tolist()

            # [4] - return the optimal covariance.
            # ====================================
            sOpt = xOpt.pop(0)
            nOpt = xOpt.pop(0)
            rOpt = xOpt.copy()
            self.sOpt = sOpt
            self.nOpt = nOpt
            self.n    = nOpt
            self.ridge= nOpt
            self.rOpt = rOpt
            self.CKM  = CKM
            return CKM

        def train_aagp(self):
            '''
            # Train AAGP
            This function will train the Adjacency-Adaptive Gaussian Process (AAGP) model using max likelihood.
            '''
            def fit_aagp(x,y,a,V_, hypersIn, returnLike = True):
                Hin = np.ravel(hypersIn).tolist().copy()
                s = Hin[0]
                n = Hin[1]
                rx= [Hin[2]]*x.shape[1]
                ra= [Hin[3]]*a.shape[1]
                rxa = rx + ra
                ckm = lambda x1,x2: self.create_kernel_matrix(x1,x2,r=rxa,s=s)
                if returnLike==False:
                    ckm = lambda x1,x2: self.create_kernel_matrix(
                        np.column_stack((x1, V_(x1))),
                        np.column_stack((x2, V_(x2))),
                        r=rxa,s=s
                        )
                    return ckm

                else:
                    xv= np.column_stack((x,a))
                    K = ckm(xv,xv)
                    H = K + self.I * n
                    m = SLOGDET(H,y)
                    return m
                


            # [1.1] - get the bounds and data
            # ===================================
            sBounds = self.sBounds
            rBounds = self.rBounds
            nBounds = self.nBounds
            x       = self.x
            y       = self.y
            xa      = self.xa

            
            # [1.2] - fit the adjacency vectorizer
            # ====================================
            AV = AdjacencyVectorizer(
                x = x,
                y = y,
                xa = xa,
                train = True,
                plot_ALE = False,
            )
            SV = AV.fit()
            a  = SV(x)

            # [2] - set the bounds
            # ===================================
            idx = 0
            bLo = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD)) + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            idx = 1
            bHi = [sBounds[idx],nBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD)) + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            optBounds = [bLo,bHi]
            
            
            
            # [3] - run the optimization
            # ====================================
            optFunc = lambda hypersIn: fit_aagp(x,y,a,SV,hypersIn, returnLike=True)
            retFunc = lambda hypersIn: fit_aagp(x,y,a,SV,hypersIn, returnLike=False)
            O,xOpt,yOpt = class_execute_optimizer(optFunc = optFunc, optBounds=optBounds, O='pso')
            CKM  = retFunc(xOpt.copy())
            xOpt = np.ravel(xOpt).tolist()
            
            # [4] - return the optimal covariance.
            # ====================================
            sOpt = xOpt.pop(0)
            nOpt = xOpt.pop(0)
            rOpt = xOpt.copy()
            self.sOpt = sOpt
            self.nOpt = nOpt
            self.n    = nOpt
            self.ridge= nOpt
            self.rOpt = rOpt
            self.CKM  = CKM
            return CKM

        def train_lod(self):
            '''
            # Train LOD
            This function will train a kernel via subsampled GCV for LOD, and return the estimator and varinace estimator of the optimal model.
            '''


            def lod_gcv(x,y,z,r=1,s=1,n=1,l=0,returnLike=True):
                '''
                # Generalized Cross-Validation for LOD
                '''
                xa = VS(x,z)
                CKM = lambda x1,x2: self.create_kernel_matrix(x1,x2, r=r, s=s, ARD=self.ARD)
                Kxs = CKM(xa,x)
                Kxx = CKM(xa,xa)
                L   = CL(Kxx, normalize=True)
                nu  = x.shape[0]/(np.square(xa.shape[0]))

                if returnLike==True:
                    Kss = Kxs[0:x.shape[0],:]
                    H = Kss + n * np.matrix(np.eye(x.shape[0]))
                    if l > 0:
                        nu= 1/x.shape[0]
                        L = nu * l * (L[0:x.shape[0],:])[:,0:x.shape[0]] @ Kss
                        H = H + L
                    HI = SLV(H)
                    e = y - Kss @ HI @ y
                    v = np.diag(Kss @ HI @ Kss.T)
                    m = np.log(np.square(e).mean()/np.square(v.mean()))
                    return m

                H   = Kxs @ Kxs.T + n * Kxx
                if l > 0:
                    H = H + nu * l * Kxx @ L @ Kxx
                HI  = np.matrix(np.linalg.pinv(H))
                B   = HI @ Kxs @ y
                mu  = lambda xIn: CKM(xIn,xa) @ B
                sig = lambda xIn: np.matrix(np.abs(np.diag(CKM(xIn,xa) @ HI @ CKM(xa,xIn)))).reshape(-1,1)
                return CKM, mu, sig

            def lod_trainer(x,y,z, hypers, returnLike=True, percentage = 0.25):
                '''
                # Laplacian Optimal Design (LOD) Trainer
                This function will train LOD using Generalized cross validation and return the optimal regression model.
                '''
                HIN = np.ravel(hypers).tolist().copy()
                s = HIN.pop(0)
                n = HIN.pop(0)
                l = HIN.pop(0)
                r = HIN.copy()
                if returnLike:
                    
                    E = []
                    nsplits = int(z.shape[0]/(z.shape[0]*percentage))
                    splits  = np.array_split(range(z.shape[0]), nsplits)
                    for i in range(len(splits)):
                        e = lod_gcv(x,y,z[splits[i],:], r=r, s=s,n=n,l=l,returnLike=True)
                        E.append(e)
                    E = np.sqrt(np.square(E).sum())
                    return E
                
                else:
                    return lod_gcv(x,y,z, r=r, s=s,n=n,l=l,returnLike=False)
            
            # [1] - get the bounds and data
            # ===================================
            sBounds = self.sBounds
            rBounds = self.rBounds
            nBounds = self.nBounds
            lBounds = GET_LAPLACIAN_BOUNDS()
            x       = self.x
            y       = self.y
            xa      = self.xa

            # [2] - set the bounds
            # ===================================
            idx = 0
            bLo = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            idx = 1
            bHi = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            optBounds = [bLo,bHi]

            # [3] - perform subsampling for faster execution
            # ================================================
            sIdx = []
            for i in range(x.shape[0]):
                xi = x[i,:]
                delta = np.ravel(np.abs(xi-xa).sum(1))
                idx = np.argmin(delta)
                if delta[idx] == 0:
                    sIdx.append(i)

            uIdx            = [g for g in range(xa.shape[0]) if not g in sIdx]
            z               = xa[uIdx,:]
            zu              = xa[uIdx,:]
            zIn             = z.copy()
            zOut            = z.copy()
            z,zu,zdx, zudx  = lhs_select(xa=zu, m=int(zu.shape[0]/5), maxIters=100, verbose=False)
            zIn             = z.copy()


            optFunc = lambda hypersIn: lod_trainer(x,y,zIn,hypersIn,returnLike=True)
            retFunc = lambda hypersIn: lod_trainer(x,y,zOut,hypersIn,returnLike=False)

            # [4] - run the optimizer
            # ===================================
            O,xOpt,yOpt = class_execute_optimizer(optFunc = optFunc, optBounds=optBounds, O='pso')
            CKM,mu,sig  = retFunc(xOpt)
            xOpt = np.ravel(xOpt).tolist()

            # [4] - return the optimal covariance.
            # ====================================
            sOpt = xOpt.pop(0)
            nOpt = xOpt.pop(0)
            rOpt = xOpt.copy()
            self.sOpt = sOpt
            self.nOpt = nOpt
            self.n    = nOpt
            self.ridge= nOpt
            self.rOpt = rOpt
            self.CKM  = CKM
            self.lapRLS_mu = mu
            self.lapRLS_sig = sig
            return CKM,mu,sig
        

        def train_lrk(self):
            '''
            # Train LRK
            This function will train a kernel via Warped RKHS and return the optimal covariance function.
            '''


            def lrk_mle(x,y,z,r=1,s=1,n=1,l=0,L = None, returnLike=True):
                '''
                # GP MLE with Warped RKHS
                '''
                if z is None:
                    xa = x.copy()
                else:
                    xa = VS(x,z)
                ckm_0 = lambda x1,x2: self.create_kernel_matrix(x1=x1, x2=x2, r=r, s=s,ARD=self.ARD)
                ckm   = lambda x1,x2: self.create_kernel_matrix(x1=x1, x2=x2, r=r, s=s, ARD=self.ARD)

                if l > 0:
                    Ka = ckm_0(xa,xa)
                    if L is None:
                        La = CL(Ka, normalize=True)
                    else:
                        La = L
                    nu = l * x.shape[0]/np.square(xa.shape[0])
                    LB = np.matrix(np.eye(Ka.shape[0])) + nu * La @ Ka
                    LB = SLV(LB,La*nu)
                    ckm = lambda x1,x2: ckm_0(x1,x2) - ckm_0(x1,xa) @ LB @ ckm_0(xa,x2)
                
                if returnLike:
                    K = ckm(x,x)
                    H = K + self.I * n
                    return SLOGDET(H, y)
                else:
                    return ckm
                



            def lrk_trainer(x,y,z, hypers, returnLike=True):
                '''
                # Trainer for GP + Warped RKHS
                This function will train LRK using GP Max Likelihood and Warmed RKHS.
                '''
                HIN = np.ravel(hypers).tolist().copy()
                s = HIN.pop(0)
                n = HIN.pop(0)
                l = HIN.pop(0)
                r = HIN.copy()
                if returnLike:
                    e = lrk_mle(x,y,z, r=r, s=s,n=n,l=l,returnLike=True)
                    return e
                else:
                    return lrk_mle(x,y,z, r=r, s=s,n=n,l=l,returnLike=False)
            
            # [1] - get the bounds and data
            # ===================================
            sBounds = self.sBounds
            rBounds = self.rBounds
            nBounds = self.nBounds
            lBounds = GET_LAPLACIAN_BOUNDS()
            x       = self.x
            y       = self.y
            xa      = self.xa

            # [2] - set the bounds
            # ===================================
            idx = 0
            bLo = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            idx = 1
            bHi = [sBounds[idx],nBounds[idx],lBounds[idx]] + [rBounds[idx]] * (x.shape[1] ** int(self.ARD))
            optBounds = [bLo,bHi]

            # [3] - perform subsampling for faster execution
            # ================================================
            sIdx = []
            for i in range(x.shape[0]):
                xi = x[i,:]
                delta = np.ravel(np.abs(xi-xa).sum(1))
                idx = np.argmin(delta)
                if delta[idx] == 0:
                    sIdx.append(i)

            uIdx            = [g for g in range(xa.shape[0]) if not g in sIdx]
            z               = xa[uIdx,:]
            zu              = xa[uIdx,:]
            zIn             = x.copy()#z.copy()
            zOut            = z.copy()
            z,zu,zdx, zudx  = lhs_select(xa=zu, m=int(zu.shape[0]/5), maxIters=100, verbose=False)
            # zIn             = z.copy()


            optFunc = lambda hypersIn: lrk_trainer(x,y,None,hypersIn,returnLike=True)
            retFunc = lambda hypersIn: lrk_trainer(x,y,zOut,hypersIn,returnLike=False)

            # [4] - run the optimizer
            # ===================================
            O,xOpt,yOpt = class_execute_optimizer(optFunc = optFunc, optBounds=optBounds, O='pso')
            CKM  = retFunc(xOpt)
            xOpt = np.ravel(xOpt).tolist()

            # [4] - return the optimal covariance.
            # ====================================
            sOpt = xOpt.pop(0)
            nOpt = xOpt.pop(0)
            rOpt = xOpt.copy()
            self.sOpt = sOpt
            self.nOpt = nOpt
            self.n    = nOpt
            self.ridge= nOpt
            self.rOpt = rOpt
            self.CKM  = CKM
            return CKM







































    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - ALGORITHMS
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """




    # [0] - package import
    # =================================
    import itertools
    import numpy as np
    from sklearn.metrics.pairwise import euclidean_distances as ED, manhattan_distances as MD

    # [1] - Developed packages
    # ==========================================
    '''
    from auxiliaries import(
        VS
    )
    '''


    # [2] - functions
    # =================================
    def maxiMin_acquisition(xRef,xIn):
        return (euclidean_distances(xRef,xIn).min(0)).reshape(-1,1)

    def lhs_sampling(n=10, p=2, seed=777, iterations=1000, sampling_mode=['distance','normsort'][0]):
        def lhs_sampler(n=n, p=p, seed=seed):
            # [INTRO] - uses uniform sampling to generate LHS samples.
            np.random.seed(seed)
            x = np.random.uniform(size=[n, p])
            for i in range(0, p):
                x[:, i] = (np.argsort(x[:, i]) + 0.5) / n
            return x

        minDist = -np.inf
        for i in range(iterations + 1):
            x = lhs_sampler(n=n, p=p, seed=seed + i)

            if 'dist' in sampling_mode:
                d = euclidean_distances(x, x)
                d = d[d>0].min()
                # d = d[d > 0]
                # d = d.min()
                if d > minDist:
                    minDist = d
                    xOut = x
            else:
                d = np.ravel(np.sqrt(np.square(x).sum(1)))
                d = d[np.argsort(d)]
                d = d[1]-d[0]
                if d > minDist:
                    minDist = d
                    xOut    = x
        return xOut


    def FFD(dims, levels):
        '''
        # FFD
        generates a full factorial design of `dims` dimensions.
        - `levels` can either be a single INT or a LIST of `dims` INTS.
        '''
        if isinstance(levels, list):
            yeets = [np.linspace(0, dims, levels[g]).tolist()
                    for g in range(len(levels))]
        else:
            yeets = [list(range(dims)) for g in range(dims)]
        return np.array(list(itertools.product(
            *[g for g in yeets]))) / np.max(yeets)


    def geoSpace(n=300,xa=None, m=10, x = None, bins = 10,p=None, verbose=False):
        '''
        # GeoSpace
        This function takes input points and selects `m` points from varying `bins` of increasing distance.
        '''
        if xa is None:

            if not x is None:
                p = x.shape[1]
            if p is None:
                print('ERROR: < p > variable for dimensions is not defined.')
            xa = lhs_sampling(n=n, p=p, iterations=5000, sampling_mode='normsort')
        xa_ = xa.copy()

        bins_ = int(np.copy(bins))

        failure = 0
        VS = lambda x1,x2: np.matrix(np.vstack((x1,x2)))
        da = euclidean_distances(xa,xa)
        da_= da # da[da>0]
        spaces = np.linspace(da_.min(), da_.max(), bins)

        if x is None:
            x = xa.min(0)
        x_ = x.copy()


        counter = 0
        while counter < m and failure <= bins:
            for i in range(len(spaces)-1):
                h0 = spaces[i]
                h1 = spaces[i+1]


                grabs = list(set(np.where((da>=h0) & (da<h1))[0].tolist()))
                xG    = xa[grabs,:]
                try:
                    d     = euclidean_distances(x,xG).min(0)
                    failure= 0
                except:
                    # print('WARNING. NO POINTS IN THIS DISTANCE: \n%s-%s: %s/%s'%(h0,h1,i+1,len(spaces)-1))
                    failure += 1
                    continue
                idx   = np.argmax(d)
                x     = VS(x,xa[idx,:])
                xa    = np.delete(xa,idx,axis=0)
                da    = np.delete(da,idx,axis=0)
                da    = np.delete(da,idx,axis=1)
                counter += 1
                if verbose==True:
                    print('%s/%s points found.'%(counter+1,m))
                # print(xs)
        if failure > bins:
            bins -= 1
            x,xa = geoSpace(xa_, m = m, x = x_, bins = bins)

        return x,xa

    def lhs_select(xa, m=500, maxIters=500, verbose=True):
        '''
        # LHS Select
        This function takes an input dataset `xa` and selects `m` points from it that best maximize the latin hypercube criterion.
        '''
        idx_tril = np.tril_indices(m,k=-1)
        def ckm(xIn):
            d = euclidean_distances(xIn)[idx_tril]
            return d[d>0].min()

        idx = np.random.choice(xa.shape[0], m, replace=False)
        idx_out = idx
        xOut= xa[idx,:]
        s0  = ckm(xOut)
        for i in range(maxIters):
            np.random.seed(777+i)
            idx   = np.random.choice(xa.shape[0], m, replace=False)
            xTest = xa[idx,:]
            s1    = ckm(xTest)

            if s1 > s0:
                s0 = s1
                xOut = xTest
                idx_out = idx

                if verbose==True:
                    print('Iteration: %s/%s | Score: %0.3f'%(i+1,maxIters, s1))

        jdx = [g for g in range(xa.shape[0]) if not g in idx_out]
        return xOut, xa[jdx,:], idx_out, jdx




    def GET_LAPLACIAN_BOUNDS():
        '''
        # Get Laplacian Bounds
        This function is a simple grabber to return consistent laplacian regularization penalty coefficient upper and lower bounds for ALL applications.
        '''
        return [0,1e-2]



    def euclidean_distances(x1,x2 = None):

        '''
        # Euclidean distances
        This computes the euclidean distances between two points, enforcing `np.array` for compatibility issues.
        '''

        if x2 is None:
            return ED(np.asarray(x1), np.asarray(x1))
        else:
            return ED(np.asarray(x1), np.asarray(x2))
        
    def manhattan_distances(x1,x2 = None):

        '''
        # Manhattan distances
        This computes the manhattan distances between two points, enforcing `np.array` for compatibility issues.
        '''

        if x2 is None:
            return MD(np.asarray(x1), np.asarray(x1))
        else:
            return MD(np.asarray(x1), np.asarray(x2))


    def SLOGDET(H, y, eps = 0.000000125):

        '''
        # SLOGDET
        This function computes the GP likelihood via spectral decomposition
        '''

        # [1] - try to perform SVD
        # ===============================
        success = True
        try:
            u,s,v = np.linalg.svd(H)


        # [2] - if it fails, then add a ridge parameter to the hessian to increase stability
        # ====================================================================================
        except:
            success = False
        
        if success == False:
            counter = 0
            while counter < 100:
                H = H + np.eye(H.shape[0]) * eps
                try:
                    u,s,v = np.linalg.svd(H)
                    success = True
                    break
                except:
                    counter += 1
            if counter == 100:
                print('[ FATAL ] - Algorithms.SLOGDET: SVD failed.')
                np.linalg.svd(H)
        
        # [3] - if it succeeds, then we can continue with the spectral decomposition.
        # ===========================================================================
        s[s==0] = 1e-20
        si      = np.diag(np.ravel(np.divide(1.0,s)))
        HI      = u @ si @ v
        sProd   = np.product(np.abs(s))

        m       = float(np.trace(y.T @ HI @ y))
        logDet  = 0
        if sProd > 0:
            try:
                logDet = np.product(np.sign(s)) * np.log(sProd)
            except:
                try:
                    logDet = np.product(np.sign(s)) * np.log(sProd)
                except:
                    logDet = np.product(np.sign(s)) * np.log(sProd)
        
        LL = m + logDet
        return LL


    def trainTest_binSplitter(my_list, n=-1, x=None):

        '''
        Bin splitter for training and testing
        '''
        n=int(np.ceil(len(my_list))*0.75)
        nSplits = n
        if n == -1:
            m = len(my_list)
            if m < 20:
                nSplits = x.shape[0]
                if m >= 20 and m < 40:
                    nSplits = 5
                if nSplits >= 40:
                    nSplits = -10


        if nSplits > 0:
            xOut = [g.tolist() for g in np.array_split(my_list,int(np.abs(nSplits)))]
        elif not x is None:
            splits = geoSpace(xa=x,m=int(x.shape[0]/10), p=x.shape[1])[0]
            splitter=[]
            for i in range(splits.shape[0]):
                idx = np.argmin(np.ravel(np.abs(splits[i,:]-x).sum(1)))
                splitter.append(idx)
            splits = [splitter] + [[g for g in my_list if not g in splitter]]
            xOut=splits
        return xOut




    def CL(K, normalize=False, walk=False, signless=False):
        '''
        # Compose Laplacian (CL)
        This function takes a symmetric kernel matrix `K` and generates the Laplacian matrix `L = S - K`, where `S` is `diag(K.sum(1))`.
        - `normalize`: normalize the laplacian (default is `False`)
        - `walk`: make random-walk laplacian (default is `False`)
        - `signless`: make signless laplacian (default is `False`)
        '''

        S = np.ravel(K.sum(1))
        SS= np.diag(S)
        if signless:
            L = SS + K
        else:
            L = SS - K
        
        if normalize or walk:
            SD = np.sqrt(S.copy())
            SD[SD==0] = 1

            D = np.diag(np.ravel(np.divide(1.0, SD)))
            if not walk:
                L = D @ L @ D
            else:
                L = D @ L
        return L


    def densityGrabber(xIn, gIn=None, nSamples=250):
        '''
        # Density grabber
        This function fits a density to `xIn` and selects a point from `gIn` using `nSamples` that have the highest density with respect to `xIn`.
        '''

        if gIn is None:
            gIn = euclidean_distances(xIn,xIn)[np.tril_indices(xIn.shape[0], k=-1)]
        
        sets = np.linspace(gIn.min(), gIn.max(), nSamples).reshape(-1,1)
        gIn  = gIn.reshape(-1,1)

        grabs = np.exp(
            -1.0 * np.square(
                euclidean_distances(sets,gIn) / np.percentile(gIn,50)
            )
        ).sum(1)

        s = sets[np.argmax(grabs),0]
        return s



    def SLV(a,b=None, safetyRidge = 0.0000125):
        '''
        # System of Equations Solver
        This function solves a system of equations `ax = b` or inverts a matrix `a` if `b=None`.
        '''
        if b is None:
            try:
                mi = np.linalg.pinv(a)
            except:
                mi = np.linalg.pinv(a + np.eye(a.shape[0]) * safetyRidge)

        else:
            try:
                mi = np.linalg.solve(a,b)
            except:
                mi = np.linalg.pinv(a) * b

        return mi




    def subsampler(x_,sampling=0.25, verbose=True):
        if sampling < 1 and sampling > 0:
            sampling = int(x_.shape[0] * sampling)
        idxs = []
        x = x_.copy()
        idx = np.argmin(np.sqrt(np.square(x).sum(1)))
        xout = x[idx,:].reshape(1,-1)
        # x = np.delete(x,idx,axis=0)
        idxs.append(idx)
        # for i in TQDM(range(sampling-1),disable=not verbose, desc='Subsampling %s points from %s'%(sampling, x_.shape[0])):
        for i in range(sampling-1):

            d = euclidean_distances(xout,x).min(0)
            idx = np.argmax(d)
            xout = np.vstack((xout,x[idx,:].reshape(1,-1)))
            idxs.append(idx)
            # x = np.delete(x,idx,axis=0)

        return xout,idxs

    def GBAL_acq(xRef, xIn, dFunc = [euclidean_distances, manhattan_distances], simple=False):
        dFunc = dFunc[int(simple)]

        scores = []
        ranger = range(xIn.shape[0])
        if simple==True:
            d0 = dFunc(xRef, xIn).min(0)
            scores = d0

        else:
            d00 = dFunc(xRef,xIn)#.min(0)
            for i in ranger:
                j  = [g for g in ranger if not g == i]

                d0 = d00[:,j].min(0)
                d1 = dFunc(VS(xRef,xIn[[i],:]), xIn[j,:]).min(0)

                d  = d0.sum()-d1.sum()
                scores.append(d)
        scores = np.matrix(scores).reshape(-1,1)
        return scores

    def mutual_euclid(xRef, xIn, xall, simple=True):
        # this function does mutual information based on euclidean distances and measured/unmeasured sets.

        scores = xIn[:,0].reshape(-1,1) * 0
        for i in range(xIn.shape[0]):
            zRef = np.delete(xIn.copy(), i,axis=0)
            xscore = GBAL_acq(np.vstack((xRef,xIn[i,:].reshape(1,-1))),xall, simple=simple).mean()
            zscore = GBAL_acq(zRef,xall, simple=simple).mean()

            scores[i,:] = float(zscore/xscore)
        return scores.reshape(-1,1)

    def GREEDY_acq(xRef, mu, xIn, dFunc = euclidean_distances, improveGreed = False):
        yRef   = mu(xRef)
        yIn    = mu(xIn)
        mode   = ['GSy', 'iGS'][int(improveGreed)]


        if mode == 'iGS':
            d0 = np.multiply(dFunc(xRef, xIn), dFunc(yRef, yIn)).min(0)
        else:
            d0 = dFunc(yRef, yIn).min(0)


        scores = d0

        scores = np.matrix(scores).reshape(-1,1)
        return scores



    def get_kernel_data_bounds(x,y,xa=None, semi_variance=False, low_noise=True):
        '''
        # Get Kernel Data Bounds
        This function grabs data-driven boundaries for kernel based models.
        '''

        # [1] - grab the lower triangle indices
        # ======================================
        trils = np.tril_indices(x.shape[0],k=-1)
        if xa is None:
            trila = trils.copy()
            xa    = x.copy()
        else:
            trila = np.tril_indices(xa.shape[0], k=-1)
        

        # [2] - compute the pairwise differences in y
        # =============================================
        v = euclidean_distances(y)
        if semi_variance:
            v = np.square(v)
        vt = v[trils]

        # [3] - grab the distances for each dimension of < x > and < xa >
        # =================================================================
        # '''
        da = np.zeros((xa.shape[0],xa.shape[0],xa.shape[1]))
        ds = np.zeros((x.shape[0],x.shape[0],x.shape[1]))
        for i in range(da.shape[2]):
            # da[:,:,i] = euclidean_distances(xa[:,[i]])
            # ds[:,:,i] = euclidean_distances(x[:,[i]])
            da[:,:,i] = euclidean_distances(xa)
            ds[:,:,i] = euclidean_distances(x)
            
        dsx = np.sqrt(np.square(ds).sum(2))
        dax = np.sqrt(np.square(da).sum(2))
        # '''

        '''
        dsx = euclidean_distances(x,x)
        dax = euclidean_distances(xa,xa)
        '''
        dst = dsx[trils]
        dat = dax[trila]


        # [4] - grab the minimum and maximum for lengthscale, signal variance, and noise variance.
        # =========================================================================================
        # '''
        rLo = [ds[:,:,i][trils] for i in range(ds.shape[2])]
        rLo = np.mean([g[g>0].min() for g in rLo])
        # '''
        '''
        rLo = dsx[dsx>0].min()
        '''
        rHi = da.max()*3

        sLo = vt[vt>0].min()
        sHi = vt.max()

        nLo = 1e-9
        nHi = 1e-3
        if not low_noise:
            nHi = densityGrabber(gIn = vt)
        rBounds = [rLo,rHi]
        sBounds = [sLo,sHi]
        nBounds = [nLo,nHi]

        return trils, trila, dsx,v, dax, dst, vt, dat, sBounds,nBounds,rBounds,rBounds


    def matern_52(x1, x2, g = None, r = 1, s = 1, ARD = False):

        if g is None:

            if ARD == True:

                g = 0
                for i in range(x1.shape[1]):

                    x1i,x2i = x1[:,i], x2[:,i]
                    g       = g + np.square(euclidean_distances(x1i,x2i) / r[i])
                g = np.sqrt(g)

            if ARD == False:

                g = euclidean_distances(x1,x2) / float(np.ravel(r)[0])
        g = np.sqrt(5) * g
        K = np.multiply(1 + g + np.square(g) / 3, np.exp(-1.0 * g))

        return K

    def gaussian(x1,x2, g = None, r=1, s=1, ARD = False):

        if g is None:

            if ARD == True:

                g = 0
                for i in range(x1.shape[1]):

                    x1i,x2i = x1[:,i], x2[:,i]
                    g       = g + np.square(euclidean_distances(x1i,x2i) / r[i])

            if ARD == False:
                g = np.square(euclidean_distances(x1,x2) / float(np.ravel(r)[0]))

        K = np.exp(-0.5 * g)

        return K

    def euclidean(x1,x2,g=None,r=1,s=1,ARD=False):
        # just a basic euclidean kernel.
        if g is None:
            g = euclidean_distances(x1,x2)
        g[g==0] = 1
        K = np.divide(1,np.square(g))
        return K


    def skewGaussian(x1,x2, r=1, s=1, k=0, ARD=False):
        r = np.ravel(r).tolist()
        k = np.ravel(k).tolist()
        if ARD==False:
            r=r[0]
            k=k[0]
        # [REFERENCE] - Quanto's answer in: https://math.stackexchange.com/questions/3334869/formula-for-skewed-bell-curve
        # [FORMULA]   - exp(-0.5 * (g/r)^2) * [1 - kg/(2r^2)]
        # if g is None:

        if ARD: # we are just going to treat every distance as g, so no need to worry about partial derivatives.
            g1 = 0
            g2 = 0
            for i in range(x1.shape[1]):
                d  = euclidean_distances(x1[:,i],x2[:,i])
                g1 = g1 + np.square(d/r[i])
                g2 = g2 + np.square(k[i] * d/(2*np.square(r[i])))
            g1 = np.sqrt(g1)
            g2 = 1-np.sqrt(g2)
        else:
            g1 = euclidean_distances(x1,x2)/r
            g2 = 1-k*g1/(2*r)
        g1 = np.exp(-0.5 * np.square(g1))
        # K = s * np.clip(np.multiply(g1, g2),0,None)
        K =  np.multiply(g1, g2)
        return K



    def myKernel(x1,x2, g = None, r =1, s = 1, k=0, ARD = False, kernel = ['matern','gaussian','skew','euclidean'][0]):
        r = np.ravel(r).tolist()
        ARD = ARD and len(r) == x1.shape[1]


        # [CONFIRMED] - See: syandana_eDist_timeTester.py for verification that this can achieve greater calculation speed for ARD.
        if not 'euclidean' in kernel and not (x1 is None or x2 is None):
            if ARD == False and len(r) == 1:
                x1 = x1/r[0]
                x2 = x2/r[0]
            else:# elif len(r)==x1.shape[1]:#(ARD == False and len(r) == x1.shape[1]) or ARD==True:
                x1 = np.divide(x1,r)
                x2 = np.divide(x2,r)
                
            ARD = False
            r=1

        if 'mat' in kernel:
            K = matern_52(x1,x2, g=g, r=r, s=1, ARD=ARD)
        if 'gau' in kernel:
            K = gaussian(x1,x2, g=g, r=r, s=1, ARD=ARD)
        if 'euc' in kernel:
            K = euclidean(x1,x2, g=g, r=1, s=1, ARD=ARD)
        if 'skew' in kernel:
            K = skewGaussian(x1,x2, r=r, s=1, k=k, ARD=ARD)

        if s <= 0:
            K = 1 - K
        return np.abs(s) * K



























    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - EQUATIONS
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """

    # [1] - python package import
    # =================================
    import numpy as np
    # import pandas as pd
    # from sklearn.metrics.pairwise import euclidean_distances
    # from pairwise_distances import euclidean_distances
    # import os

    # [2] - self-made packages
    # =================================
    '''
    from algorithms import lhs_sampling
    '''
    cutta = lambda lo, hi, dims: [[lo for g in range(dims)], [hi for g in range(dims)]]


    def testFunctions(fName):
        # from scipy.special import factorial
        "[ INTRO ] - This is to conglomerate as many functions as I can and give them multi-dimensional support between the boundaries [-10,10]"
        "[ NOTE ] - The inputs x MUST BE BETWEEN [-10,10] FOR THE FUNCTIONS TO WORK."

        ## [ LIST OF FUNCTIONS ]
        ##-------------------------------------
        # [5] - Qing          : http://benchmarkfcns.xyz/benchmarkfcns/qingfcn.html

        ##[ FUNCTION BOUNDARIES ]
        ##--------------------------------------
        # [5] - Qing          : x = [-2,2]


        '  [ LIST OF N-Dimensional FUNCTIONS ] '
        '-------------------------------------'
        # [z.1] - Cosine Mixture
        # [z.2] - Zackarov
        # [z.4] - Rastrigin
        # [z.5] - Qing
        # [z.8] - Levy03
        # [z.3]- Schwefel20
        # [z.13] - Trid

        cutta = lambda lo, hi, dims: [[lo for g in range(dims)], [hi for g in range(dims)]]

        def convertBounds(x, bounds):
            bLo, bHi = bounds
            xg = (x+10)/20
            # xg = np.matrix(np.copy(x))
            h = 0
            for g in range(x.shape[1]):
                if h == len(bLo):
                    h = 0
                xgmin,xgmax = xg[:,g].min(),xg[:,g].min()

                # xg[:,g] = (xg[:,g] - xgmin)/(xgmax-xgmin)
                blo     = bLo[h]
                bhi     = bHi[h]
                xg[:,g] = xg[:,g] * (bhi - blo) + blo
                h += 1
            return(xg)
        
        def f1_cosine(xIn, bounds):

            x = convertBounds(xIn,bounds)
            y1= 0
            y2= 0
            for i in range(x.shape[1]):
                xi = x[:,i]
                y1 = y1 + np.cos(5.0*np.pi*xi)
                y2 = y2 - np.square(xi)
            y = 10 * (0.1 * y1 + y2)
            return y

        def f5_qing(xIn,bounds):
            x = convertBounds(xIn,bounds)
            y = 0
            for i in range(x.shape[1]):
                xi = x[:,i]
                y = y + np.square(np.square(xi)-(i+1))
            return y
        
        def f12_mishraBird(x, bounds): ## mishra's bird
            x = convertBounds(x, bounds)
            y = 0

            for i in range(x.shape[1]-1):
                j = i+1

                xi = x[:,i]
                xj = x[:,j]

                y1 = np.multiply(np.sin(xi), np.exp(np.square(1-np.cos(xj))))
                y2 = np.multiply(np.cos(xj), np.exp(np.square(1-np.sin(xi))))
                y3 = np.square(xi-xj)

                y  = y + y1 + y2 + y3
            return y


        fSplit = fName.split('.')
        dims   = int(fSplit[-1])
        f      = int(fSplit[1])


        funcBounds = {
                5: [f5_qing, [-2,2]],
                12: [f12_mishraBird, [-2*np.pi,2*np.pi]],
                1: [f1_cosine, [-1,1]],
                }
        
        func, bounds = funcBounds[f]
        boundsIn = cutta(bounds[0],bounds[1], dims)
        FUNC   = lambda input: func(np.matrix(input).reshape(-1,dims), boundsIn)
        bounds_ = cutta(-10,10, dims)


        return FUNC, bounds_


    def get_case_study_name(fName, addDim = 0):

        fSplit = fName.split('.')
        f      = fSplit[0]+'.'+fSplit[1]

        fz = {
            'z.1': 'Cosine',
            }
        fz = {
            'z.5': 'Qing',
            }
        fz = {
            'z.12': 'Bird',
            }

        addDim = addDim * ((not 'c' in fName) or 'eyelink' in fz[f])
        try:
            nameOut = fz[f]+['-%sD'%(fName.split('.')[-1]) if addDim == True else ''][0]
        except:
            nameOut = fz[fName]+['-%sD'%(fName.split('.')[-2]) if addDim == True else ''][0]
        return nameOut






    def get_function(fName, n=300, e=1000, lhs_iters = 250, structure_noise=0.075, scale=0.5):
        
        if fName == 'mlp_mnist':
            import pandas as pd
            DF = pd.read_csv('mlp_trainer_mnist_4layers.csv')
            return DF


        # [INTRO] - this grabs the functions we want from < testFunctions >.
        f, b = testFunctions(fName)
        p    = len(b[0])

        # print('generating training points.')
        xa   = lhs_sampling(n=n,p=p, seed=777, iterations=lhs_iters)

        # print('generating evaluation points.')
        xe   = lhs_sampling(n=e,p=p, seed=777, iterations=lhs_iters)

        bHi = b[1]
        bLo = b[0]
        for i in range(xa.shape[1]):
            xg = xa[:,i]
            xg = (xg - np.min(xg))/(xg.max()-xg.min()) * (bHi[i] - bLo[i]) + bLo[i]
            xa[:,i] = xg


            xg = xe[:,i]
            xg = (xg - np.min(xg))/(xg.max()-xg.min()) * (bHi[i] - bLo[i]) + bLo[i]
            xe[:,i] = xg

        ya   = f(xa)
        ye   = f(xe)

        return [np.asarray(g) for g in [xa, ya, xe, ye]] +  [f, b]




























    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - SIMULATION
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """


    # [1] - python package import
    # ==================================================
    import numpy as np
    import pandas as pd
    from joblib import Parallel, delayed
    from sklearn.metrics import r2_score
    import warnings


    # [2] - self-made package import
    # ==================================================
    '''
    from equations import get_function
    from modeling import(
        model_master,
        calculate_regression_metrics
    )
    from auxiliaries import ProgressParallel
    '''

    # [3] - functions
    # ==================================================



    def DOE(
            xa,ya,
            xe,ye,
            seed_idx,
            n_find = 30,
            model  = 'gp',
            acquisition = None,
            seed        = 0,
            verbose     = False,
    ):
        '''
        # Run DOE
        This function executes a DOE using the `model` and `acquisition` via `model_master`.
        '''
        import timeit
        defTime = lambda: timeit.default_timer()
        idx = seed_idx.copy()
        jdx = [g for g in range(xa.shape[0]) if not g in idx]

        xs  = xa[idx,:]
        ys  = ya[idx,:]
        xz  = xa[jdx,:]
        yz  = ya[jdx,:]



        _metrics = ['r2', 'mae', 'mse', 'rmse','nrmse', 'mape', 'mmape', 'wmape', 'smape','bias', 'adjusted_r2', 'cod', 'mbd', 'cv']
        biases = []
        rmses  = []
        wmapes = []
        smapes = []
        R2s = []
        fit_times = []

        metrics = {}
        for g in _metrics:
            metrics[g] = []

        for i in range(n_find+1):

                
            # [1] - fit the model
            # =================================
            xa = np.vstack((xs,xz))

            model_timer = defTime()
            mu,sig = model_master(
                x = xs,
                y = ys,
                xa= xa,
                model_name = model,
                acquisition = acquisition,
            )

            # [2] - calcualte the performance metrics
            # =======================================
            model_timer = defTime()-model_timer

            yp = np.ravel(mu(xe))
            yt = np.ravel(ye)

            for _m_ in _metrics:
                __m__ = calculate_regression_metrics(
                    y_true = yt,
                    y_pred = yp,
                    p      = xs.shape[1],
                    metric = _m_
                )
                metrics[_m_].append(__m__)

            # yp = np.ravel(mu(xe))
            # error= np.ravel(ye) - yp
            # # overEst = np.clip(error,0,None)
            # # underEst= np.clip(error,None,0)*-1

            # # overEst = np.where(error>0, np.abs(error),0)
            # # underEst= np.where(error<=0, np.abs(error),0)
            # # bias    = np.divide(overEst-underEst,overEst+underEst).mean()
            # # wmape   = np.abs(error).sum()/np.abs(ye).sum()
            # # rmse    = np.sqrt(np.square(error).mean())
            # # smape   = np.divide(
            # #                     np.ravel(np.abs(error)),
            # #                     (np.abs(np.ravel(yp)) + np.abs(np.ravel(ye)))/2
            # #                     ).mean()
            # # R2 = r2_score(np.ravel(ye), np.ravel(yp))

            # biases.append(bias)
            # wmapes.append(wmape)
            # rmses.append(rmse)
            # smapes.append(smape)
            # R2s.append(R2)


            fit_times.append(model_timer)

            # [3] - select the next optimal point to test
            # ==============================================
            vhat = sig(xz)
            sdx  = np.argmax(np.ravel(vhat))
            
            xOpt = xz[[sdx],:]
            yOpt = yz[[sdx],:]
            xs   = np.vstack((xs,xOpt))
            ys   = np.vstack((ys,yOpt))
            xz   = np.delete(xz,sdx,axis=0)
            yz   = np.delete(yz,sdx,axis=0)
            if verbose:
                _nrmse = '%0.3f'%(metrics['nrmse'][-1])
                _wmape = '%0.3f'%(metrics['wmape'][-1])
                print(f'{model} - {i}/{n_find} | nrmse: {_nrmse} | wmape: {_wmape}')
            
            del mu
            del sig
            gc.collect()
        
        # [3] - create the pandas dataframe to show the results
        # ======================================================
        # df = pd.DataFrame(
        #         {
        #             'WMAPE':wmapes,
        #             'SMAPE':wmapes,
        #             'R2':R2s,
        #             'RMSE':rmses,
        #             'Bias':biases,
        #             'Fit Time':fit_times
        #         }
        #     )
        df                  = pd.DataFrame()
        df['Samples Added'] = [g for g in range(n_find+1)]
        df['Model'] = model
        df['Acquisition'] = acquisition.upper() if not acquisition is None else 'Default'
        df['Initial Points'] = len(idx)/xa.shape[1]
        df['Total Population'] = xa.shape[0]
        df['No. Variables'] = xa.shape[1]
        df['Seed'] = seed
        df['Fit Time'] = fit_times
        for metric in list(metrics.keys()):
            m = metric
            if 'adjusted_r2' in m:
                m = 'a-r2'
            df[m.upper()] = metrics[metric]

        # df['mu'] = mu
        # df['sig']= sig
        return df

    def prepare_doe(
            models= ['gp','aagp','lod','lrk','xgboost','localridge','deepgp'],
            inits = [4,8,10],
            pops  = [200,350,500],
            replicates = 20,
            noise = 1, # percent value, do not convert to decimal.
            test_function = 'z.15.2',
    ):
        seed_counter = []
        seed_indices = []
        seed_noises  = []
        modelouts    = []
        inits_counter= []
        pops_counter = []
        XA = []
        YA = []
        XE = []
        YE = []
        # for model in TQDM(models, colour='red', bar_format="{l_bar}{bar} | {n}/{total}"):
        for model in models:
            for T in pops:
                xa, ya, xe, ye, f, b = get_function(
                                                    fName = test_function,
                                                    n = T,e=1000,lhs_iters=10,
                                                )
                np.random.seed(777)
                for I in inits:
                    m     = I * xa.shape[1]
                    seeds      = [np.random.choice(xa.shape[0],m,replace=False) for g in range(replicates)]
                    seedNoises = (np.random.rand(ya.shape[0],replicates) * 2 - 1) * noise * (ya.max()-ya.min())/100
                    counter = 0
                    # print(seeds)
                    # print(np.column_stack((ya,seedNoises[:,[0]])))
                    # import sys
                    # sys.exit('test')
                    for g in range(replicates):
                        # seed = np.random.choice(xa.shape[0],m,replace=False)
                        # nois = (np.random.rand(ya.shape[0],1) * 2 - 1) * noise * (ya.max() - ya.min())/100


                        # print(model,T,I,seed)
                        seed_indices.append(seeds[g])
                        seed_noises.append(seedNoises[:,[g]] + ya)
                        YA.append(seedNoises[:,[g]] + ya)
                        XA.append(xa)
                        YE.append(ye)
                        XE.append(xe)
                        seed_counter.append(counter)
                        modelouts.append(model)
                        pops_counter.append(T)
                        inits_counter.append(I)
                        counter+=1

        # sys.exit('test'):
        # for i in range(len(seed_indices)):
        #     print('*******************************************')
        #     print('Total Population: %s'%(XA[i].shape[0]))
        #     print('Initial Points: %s'%(len(seed_indices[i])/XA[i].shape[1]))
        #     print(seed_indices[i])
        # import sys
        # sys.exit('simulation.py line 174 checking the seeds and points.')
        return modelouts,XA,YA,XE,YE,seed_indices, seed_counter, pops_counter,inits_counter

    def run_doe(
        test_function = 'z.15.2',
        models      = ['gp','aagp','lod','lrk','xgboost','localridge','deepgp'],
        acquisition = None,
        inits       = [4,8,10],
        pops        = [200,350,500],
        replicates  = 1,
        n_find      = 30,
        noise       = 1, # percent value, do not convert to decimal.,
        parallel    = True,
        nJobs       = -1
    ):

        M,XA,YA,XE,YE,SI,SC,TC,IC      = prepare_doe(test_function=test_function,models=models,inits=inits, pops=pops, replicates=replicates, noise=noise)
        print('Prepping Models.')
        if acquisition is None:
            acquisition = []
            for g in models:
                if '-' in g:
                    # print('Populating Acquisition: ', g.split('-'))
                    acquisition.append(g.split('-')[-1])
                else:
                    acquisition.append(None)


        APP             = lambda iii: DOE(
            xa          = XA[iii],
            ya          = YA[iii],
            xe          = XE[iii],
            ye          = YE[iii],
            model       = M[iii],
            acquisition = acquisition[ models.index(M[iii]) ],
            seed        = SC[iii],
            seed_idx    = SI[iii],

            n_find      = n_find,
        )

        print('========================================================================')
        print('Running Simulation. Total Processes: %s |'%(len(M)), '%s Jobs: '%('Parallel' if parallel else 'Series'), nJobs)
        print('========================================================================')
        with warnings.catch_warnings():
            np.seterr('ignore')
            warnings.simplefilter('ignore')
            if parallel:
                DF = pd.concat(
                    # Parallel(n_jobs = nJobs)(delayed(APP)(i) for i in range(len(M))),
                    ProgressParallel(n_jobs = nJobs, use_tqdm=True, timeout=None, backend='loky', total=len(M))(delayed(APP)(i) for i in range(len(M))),

                    axis=0,
                    ignore_index=True
                )
                
            else:
                DF = pd.concat(
                    [APP(i) for i in range(len(M))], axis=0, ignore_index=True
                )
        DF['Test Function'] = test_function
        return DF,XA,YA

































    """
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    SECTION - EXECUTOR
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    ======================================================
    """
    # [1] - import the required packages
    # =============================================
    import os
    '''
    import SETTINGS
    SETTINGS.install_packages()
    '''

    # install_packages()



    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from timeit import default_timer as defTime
    '''
    from visualization import SETSTYLE
    from simulation import run_doe
    '''
    SETSTYLE('bmh', clear=True)
    current_directory = os.path.dirname(os.path.abspath(__file__))

    # [2] - set up and run the simulation
    # ==============================================
    print()
    print('Running simulation in directory ---->', current_directory)
    test_function_title = 'Qing (3D)' if 'z.5' in test_function.lower() else 'Cosine (10D)'
    print('Test Function  ---> ',test_function_title)
    models_1 = [g for g in models if not 'deepgp' in g.lower()]
    models_2 = [g for g in models if not g in models_1]
    print('Group 1 models ---> ', models_1)
    print('Group 2 models ---> ', models_2)
    tstart = defTime()

    # [3] - set up the simulation parameters
    # ========================================
    simulation_parameters = dict(
                    test_function   = test_function,
                    inits           = inits,
                    pops            = pops,
                    noise           = noise,
                    n_find          = n_find,
                    replicates      = replicates,
                    parallel        = parallel,
                    nJobs           = n_jobs,
    )

    # [4] - run group 1 in parallel
    # ==============================================
    DF_1 = pd.DataFrame()
    if True:
        simulation_parameters['models'] = models_1
        DF_1,XA,YA = run_doe(**simulation_parameters)

    tend1 = defTime()
    tend = tend1-tstart
    print('Group 1 completed. Time   : %0.2fs (%0.2fmins, %0.2fhrs)'%(tend,tend/60,tend/3600))

    # [5] - run group 2 in series (memory issue)
    # ==============================================
    DF_2 = pd.DataFrame()
    if True:
        simulation_parameters['models'] = models_2
        
        # change for low-RAM machines
        simulation_parameters['nJobs'] = int(n_jobs/2)
        DF_2,XA,YA = run_doe(**simulation_parameters)
    tend2 = defTime()
    tend = tend2-tend1
    tender = tend2 - tstart
    print('Group 2 completed. Time   : %0.2fs (%0.2fmins, %0.2fhrs)'%(tend,tend/60,tend/3600))
    print('All groups completed. Time: %0.2fs (%0.2fmins, %0.2fhrs)'%(tender,tender/60,tender/3600))


    # [6] - Concatenate the results, visualize, and output
    # ======================================================
    DF = pd.concat((DF_1,DF_2),axis=0,ignore_index=True)
    DF['Model'] = DF['Model'].replace(to_replace=renames)
    agg = DF.groupby(['Model','Samples Added']).agg({'NRMSE':'mean'}).reset_index()
    fig = plt.figure(figsize=(8,5),dpi=350)
    ax = fig.add_subplot(1,1,1)
    sns.lineplot(
        data = agg,
        x = 'Samples Added',
        y = 'NRMSE',
        hue = 'Model',
        palette = model_palette,
        linewidth=2
    )
    ax.set_title(test_function_title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.5,1), frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig(current_directory + '/Figure 3.jpg',dpi=350)
    print('Output JPG saved to ---->', current_directory + '/' + 'Figure 3.jpg')

if os.path.basename(__file__) == 'temp_exe.py':
    RUN_PIPELINE()

#### - STOP PYTHON


def extract_section(input_file, output_file, start_marker, stop_marker):
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()

    start_index = None
    stop_index = None

    for i, line in enumerate(lines):
        if start_marker in line:
            start_index = i
        if stop_marker in line:
            stop_index = i
            break

    if start_index is not None and stop_index is not None:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.writelines(lines[start_index:stop_index + 1])
    else:
        print("Markers not found in the file.")




if __name__ == "__main__":
        
    # Example usageos.path.basename(__file__)
    input_file = os.path.basename(__file__)
    output_file = 'temp_exe.py'
    start_marker = "### - START PYTHON"
    stop_marker = "### - STOP PYTHON"
    extract_section(input_file, output_file, start_marker, stop_marker)

    # [1] - grab the directories
    host_folder=os.path.dirname(os.path.abspath(__file__))
    venv_folder= f'{host_folder}/venv'
    try:
        os.mkdir(venv_folder)
    except:
        pass
    print('\n'*2)
    print(f'Current directory: {host_folder}')
    print(f'Venv directory   : {venv_folder}')

    # [2] - make a venv
    vb = venv.EnvBuilder(
        system_site_packages=False,
        clear=True,
        symlinks=False,
        with_pip=True
    )
    vb.create(env_dir=venv_folder)

    # [3] - set the desired file
    exe_script = f'"{host_folder}/{output_file}"'
    print(f'Running          : {exe_script}')

    # [4] - Determine activation command based on OS
    if platform.system() == 'Windows':
        activate_cmd = os.path.join(venv_folder, 'Scripts', 'activate')
    else:
        activate_cmd = os.path.join(venv_folder, 'bin', 'activate')
    activate_cmd = f'"{activate_cmd}"'

    # [5] - Combine activation with running the script
    print(f'Activate command : {activate_cmd}')
    if platform.system() == 'Windows':
        command = f"{activate_cmd} && python {exe_script}"
        subprocess.run(command, shell=True)
    else:
        command = f"source {activate_cmd} && python {exe_script}"
        subprocess.run(command, shell=True, executable="/bin/bash")
    os.remove(output_file)
