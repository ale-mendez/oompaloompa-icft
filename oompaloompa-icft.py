import numpy as np
import pandas as pd
import yaml
import os

def load_input():
    ''' 
    
    Read data required to write ICFT inputs 
    
      element     : element name (H, He, Li, Be, ...)
      nelec       : number of electrons 
      mpot        : type of model potencial (STO, TFDA)
      ppot        : whether polarization potencial is used 
      alfd        : alpha parameter 
      rcut        : rho parameter 
      jmax_ex     : maximum value of J for exchange calculation
      jmax_nx     : maximum value of J for non exchange calculation
      maxc        : maximum number of partial waves
      maxe        : maximum scattering energy value
      nproc       : number of processor that will be used
      eion        : ionization energy (rydbergs)
      grid_pts    : number of points used in mesh grid (several grids can be inputed)
      grid_ener   : energy values corresponding to mesh grids (equal elements than grid_pts)
      cfgs_spect  : list of configurations in spectroscopic notation
      scaling     : whether to use orbital scaling (bool)
      pseudo      : whether pseudostates are considered (bool)
      psorb       : first pseudoorbital considered 
      nshift      : number of spectroscopic terms to be shifted in stg3
      
    Other variables
    
      orbs_spect  : orbitals in spectroscopic notation
      
      element     : element dictionary containing
                    'name'   - element name
                    'zcharge' - nuclear charge
                    'nelec'  - number of electron
                    'eion'   - ionization energy (rydbergs)
      orbs        : orbitals dictionary containing 
                    'spect'  - spectroscopic notation 
                    'qnumbs' - quantum numbers [nq,lq]
                    'lambdas' - scaling parameter
                    'norb'   - number of orbitals
      cfgs        : electronic configuration dictionary containig
                    'spect'  - spectroscopic notation 
                    'ncfg'  - number of configurations
                    'ne_cfgs' - occupation numbers of N-electron system
                    'np1_cfgs' - occupation numbers of (N+1)-electron system
      psorbitals  : pseudoorbital dictionary contining
                    'pseudo' - boolean
                    'psorb'  - spectroscopic notation of first pseudoorbital
                    'ipsobr' - orbital index of first pseudoorbital

      datainp     : dictionary with relevant input data
      
    '''
    global datainp
    inpfname=open('icftparam.yml','r')
    data=yaml.load(inpfname,Loader=yaml.FullLoader)
    name=data['element']
    assert type(name)==str,'Variable "element" is a string with the element name. For example, for hydrogen use "H".'
    nelec=data['nelec']
    assert type(nelec)==int,'Variable "nelec" is an integer number. For example, for O V nelec is 4.'
    mpot=data['mpot']
    assert (mpot=='STO') or (mpot=='TFDA'),'Variable "mpot" values are only "STO" or "TFDA".'
    ppot=[data['ppot']]
    assert type(ppot[0])==bool,'Variable "ppot" is boolean.'
    if ppot[0]==True:
        alfd=data['alfd']
        rcut=data['rcut']
        polparam=dict(zip(['alfd','rcut'],[alfd,rcut]))
        ppot.append(polparam)
    jmax_ex=data['jmax_ex']
    jmax_nx=data['jmax_nx']
    maxc=data['maxc']
    maxe=data['maxe']
    nproc=data['nproc']
    eion=data['eion']
    grid_pts=data['grid_pts']
    grid_ener=data['grid_ener']
    cfgs_spect=data['cfgs']
    ncfg=len(cfgs_spect)
    orbs_spect=orbitals(cfgs_spect)
    norb=len(orbs_spect)
    scaling=data['scaling']
    if scaling==True: 
        lambdas=data['lambdas']
        if len(lambdas)!=norb: RaiseError('Number of "lambdas" should be equal to the number of orbitals.')
    else:
        lambdas=np.zeros(norb)+1
    pseudo=data['pseudo']
    assert type(pseudo)==bool,'Variable "pseudo" is boolean.'
    if pseudo==True: 
        psorb=data['psorb']
        assert type(psorb)==str,'Variable "psorb" is a string corresponding to the first pseudoorbital used.'
        ipsorb=flag_pseudoorbitals(lambdas,orbs_spect,norb)
    else:
        psorb=None
        ipsorb=None
    nshift=data['nshift']

    zcharge=nuclear_charge(name)
    element=dict(zip(['name','zcharge','nelec','eion'],[name,zcharge,nelec,eion]))
    nq,lq=spect2quantnumb(orbs_spect,norb)
    qnumbs=[[nq[i],lq[i]] for i in range(norb)]
    orbs=dict(zip(['spect','qnumbs','lambdas','norb'],[orbs_spect,qnumbs,lambdas,norb]))
    ne_cfgs=occ_Nsystem(cfgs_spect,orbs_spect,ncfg,norb)
    np1_cfgs=occ_Np1system(ne_cfgs,qnumbs,ncfg,norb)
    cfgs=dict(zip(['spect','ncfg','ne_cfgs','np1_cfgs'],[cfgs_spect,ncfg,ne_cfgs,np1_cfgs]))
    psorbitals=dict(zip(['pseudo','psorb','ipsorb'],[pseudo,psorb,ipsorb]))
    keys=['element','mpot','ppot','jmax_ex','jmax_nx','maxc','maxe','nproc',
          'grid_pts','grid_ener','cfgs','orbs',
          'psorbitals','nshift']
    values=[element,mpot,ppot,jmax_ex,jmax_nx,maxc,maxe,nproc,grid_pts,grid_ener,
            cfgs,orbs,psorbitals,nshift]
    datainp=dict(zip(keys, values))
    print('Input data load OK.')
    return

def nuclear_charge(name):
    ''' Determine element nuclear charge. '''
    elements=['H','He',
              'Li','Be','B','C','N','O','F','Ne',
              'Na','Mg','Al','Si','P','S','Cl','Ar',
              'K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga','Ge','As','Se','Br','Kr']
    zcharge=elements.index(name)+1
    return zcharge

def orbitals(cfgs):
    ''' Recognize the spectroscopic orbitals used in the spectroscopic configurations "cfgs". 
        **Warning** Only works for orbitals with less than 10 electrons per orbital.'''
    jcfgs=' '.join(cfgs)
    scfgs=jcfgs.split()
    mcfgs=[scfgs[i][:-1] for i in range(len(scfgs))]
    orbs=[]
    [orbs.append(x) for x in mcfgs if x not in orbs]
    return orbs

def angular_momentum():
    ''' Returns dictionary with angular momenta values '''
    lq_spect=['s','p','d','f','g','h','i','k']
    lq_numb=[0,1,2,3,4,5,6,7]
    return dict(zip(lq_spect,lq_numb))

def spect2quantnumb(orbs,norb):
    ''' Convert atomic orbitals with spectroscopic notation "orbs" to n,l quantum numbers. '''
    nq=[int(orbs[i][:-1]) for i in range(norb)]
    lspect=[orbs[i][-1] for i in range(norb)]
    L=angular_momentum()
    lq=[L[i] for i in lspect]
    return nq,lq

def occ_Nsystem(cfgs,orbs,ncfg,norb):
    ''' Determines the N electron system (number of electrons per orbital) for the 
        spectroscopic configurations given by "cfgs". '''
    icfgs=[cfgs[i].split() for i in range(ncfg)]
    ne_cfgs=[]
    for i in range(ncfg):
        ne=[0]*norb
        for j in icfgs[i]:
            for k in range(norb):
                if j[:-1]==orbs[k]: 
                    ne[k]=int(j[-1])
        ne_cfgs.append(ne)
    return ne_cfgs

def occ_Np1system(ne_cfgs,qnumbs,ncfg,norb):
    ''' Returns the N+1 electron system of the N electron system given by "ne_cfgs". '''
    np1=[]
    for i in range(ncfg):
        for j in range(norb):
            ne=ne_cfgs[i].copy()
            neocc=ne[j]
            np1occ=neocc+1
            lq=qnumbs[j][1]
            iocc=orb_occupancy(lq,np1occ)
            if iocc==True: 
                ne[j]=np1occ
                if ne not in np1:
                    np1.append(ne)
    return np1

def eocc_by_angularmomenta():
    ''' Dictionary of maximum electron occupancy according angular momenta. '''
    lq_numb=list(angular_momentum().values())
    occ=[2,6,10,14,14,14,14,14]
    return dict(zip(lq_numb,occ))

def orb_occupancy(lq,np1):
    ''' Determines whether the orbital with angular momenta lq can be occupied with np1 electrons. '''
    L=eocc_by_angularmomenta()
    if np1>L[lq]: 
        return False
    return True

def flag_pseudoorbitals(lambdas,orbs,norb):
    ''' Gives the orbital index of first pseudoorbital. '''
    ipsorb=orbs.index('5s')
    for i in range(ipsorb,norb): lambdas[i]=-lambdas[i]
    return ipsorb

def flag_modelpot(mpot):
    ''' Model potential flag for SMINIM namelist (das file). '''
    flagpot=1
    if mpot=='STO': flagpot=-flagpot
    return flagpot

def flag_orthogonality(pseudo):
    ''' Orthogonality flag for SMINIM namelist (das file). '''
    orthog='YES'
    if pseudo==True: orthog="'LPS'"
    return orthog

def oompaloompa_header(fname):
    if fname=='das': flag='A.S.'
    if fname!='das': flag='S.S.'
    return flag+' automatically generated with python script'

def namelist_salgeb():
    rad="'NO'"
    cup="'ICM'"
    norb=datainp['orbs']['norb']
    ncfg=datainp['cfgs']['ncfg']
    nmetaj=1
    nmeta=1    
    salgeb=["&SALGEB","RAD="+rad,"CUP="+cup,
            "MXVORB="+str(norb),"MXCONF="+str(ncfg),
            "KUTSO=0","NMETAJ="+str(nmetaj),"NMETA="+str(nmeta),"&END"]
    return ' '.join(salgeb)

def namelist_sminim():
    mpot=datainp['mpot']
    flagpot=flag_modelpot(mpot)
    zcharge=datainp['element']['zcharge']
    maxe=datainp['maxe']
    norb=datainp['orbs']['norb']
    ncfg=datainp['cfgs']['ncfg']
    ppot=datainp['ppot'][0]
    if ppot==True:
        alfd=datainp['ppot'][1]['alfd']
        rcut=datainp['ppot'][1]['rcut']
    pseudo=datainp['psorbitals']['pseudo']
    orthog=flag_orthogonality(pseudo)
    pprint="'FORM'"
    radout="'YES'"
    sminim=["&SMINIM","NZION="+str(flagpot*zcharge),"MAXE="+str(maxe),"NLAM="+str(norb)]
    if flagpot<0: sminim+=["MCFMX="+str(ncfg),"MEXPOT=1"]
    if ppot==True:
        fa='ALFD='
        fr='RCUT='
        for i in range(3):
            fa+=str(alfd[i])+','
            fr+=str(rcut[i])+','
        sminim+=[fa[:-1],fr[:-1]]
    if pseudo==True: sminim+=["ORTHOG="+str(orthog)]
    sminim+=["PRINT="+pprint,"RADOUT="+radout,"&END"]
    return ' '.join(sminim)

def write_orbital_qnumbs(file):
    qnumbs=datainp['orbs']['qnumbs']
    jj=2*' '
    for j in qnumbs:
        jj+=str(j[0])+' '+str(j[1])+2*' '
    file.writelines([jj,'\n'])
    return

def write_esystem(key,file,fname):
    oneblank=' '
    spacing=oneblank
    ending=''
    if fname=='dstg2': ending=' 0'
    if key=='N': ne_cfgs=datainp['cfgs']['ne_cfgs']
    if key=='N+1': ne_cfgs=datainp['cfgs']['np1_cfgs']
    ncfg=len(ne_cfgs)
    for i in range(ncfg):
        jj=oneblank
        if fname=='das': jj=3*oneblank
        for j in ne_cfgs[i]:
            if fname=='das': spacing=4*oneblank
            jj+=str(j)+spacing
        file.writelines([jj+ending,'\n'])    
    return

def write_lambdas(file):
    oneblank=' '
    lambdas=datainp['orbs']['lambdas']
    jj=oneblank
    for i in lambdas: 
        jj+=str(i)+2*oneblank
    file.writelines([jj,'\n'])
    return

def write_MCFMX(file):
    oneblank=' '
    ncfg=datainp['cfgs']['ncfg']
    mpot=datainp['mpot']
    if mpot=='STO':
        jj=oneblank
        for i in range(ncfg): 
            jj+=str(i+1)+oneblank
        file.write(jj)
    return

def write_das():
    ''' Write AutoStructure input file. '''
    fname='das'
    header=oompaloompa_header(fname)
    salgeb=namelist_salgeb()
    sminim=namelist_sminim()
    with open(fname,'w+') as das:
        das.writelines([header,'\n'])
        das.writelines([salgeb,'\n'])
        write_orbital_qnumbs(das)
        write_esystem('N',das,fname)
        das.writelines([sminim,'\n'])
        write_lambdas(das)
        write_MCFMX(das)
    print('Write '+fname+' OK')
    return

def run_AS():
    try:
        os.system('~/AS/asdeck25.x < das')
        print('AS run OK.')
    except OSError:
        print('Error running AutoStructure... Check das!')
    return

def load_NIST():
    ''' Read NIST input data. '''
    global NIST
    try:
        terNIST=pd.read_csv('NIST_terms.dat',sep='\s+',skiprows=[0,2])
        change_cfgs_format(terNIST)
        levNIST=pd.read_csv('NIST_levels.dat',sep='\s+',skiprows=[0,2])
        change_cfgs_format(levNIST)
    except:
        print('NIST_level or NIST_terms files not found')
        levNIST=None
        terNIST=None
    NIST=dict(zip(['Terms','Levels'],[terNIST,levNIST]))
    print('NIST data load OK.')
    return

def change_cfgs_format(nist_df):
    old_format=nist_df['Configuration'].tolist()
    new_format=[]
    L=angular_momentum().keys()
    for i in old_format:
        cfg_split=i.split('.')
        i=0
        for j in cfg_split:
            if j[-1] in L: cfg_split[i]+='1'
            i+=1
        new_format.append(' '.join(cfg_split))
    nist_df['Configuration']=new_format
    return

def load_AS():
    global AS,NIST
    as_terms,etot=load_df('TERMS')
    as_levels,etot=load_df('LEVELS')
    include_NIST(as_terms,NIST['Terms'])
    include_NIST(as_levels,NIST['Levels'])
    AS=dict(zip(['Terms','Levels','Etot'],[as_terms,as_levels,etot]))
    print('AS data load OK.')
    return

def load_df(asfile):
    try: 
        asdf=pd.read_csv(asfile,sep='\s+')
        etot=get_Etotal(asdf)
        include_cfg_spect(asdf)
        include_term_spect(asdf)
    except OSError:
        print("OS error: Error reading {}... Check olg!".format(asfile))
    return asdf,etot

def get_Etotal(asdf):
    nrows=len(asdf)-1
    etot=asdf.iloc[nrows]['ENERGY(RYD)']
    asdf.drop(nrows,axis=0,inplace=True)
    return etot

def include_cfg_spect(asdf):
    configuration=[]
    cols=asdf.columns.tolist()
    for i in asdf['CF']:
        configuration.append(datainp['cfgs']['spect'][i-1])
    if 'Configuration' not in cols: asdf.insert(0,'Configuration',configuration)
    if 'CF' in cols: asdf.drop('CF',axis=1,inplace=True)
    return

def include_term_spect(asdf):
    nrows=len(asdf)
    # multiplicity
    int_multiplicity=asdf['S'].tolist()
    str_multiplicity=[str(i) for i in int_multiplicity]
    # angular momentum
    L=[i.capitalize() for i in angular_momentum().keys()]
    int_angmomentum=asdf['L'].tolist()
    str_angmomentum=[L[i] for i in int_angmomentum]
    # parity
    P=['','*']
    int_parity=asdf['P'].tolist()
    str_parity=[P[i] for i in int_parity]
    spect_term=[str_multiplicity[i]+str_angmomentum[i]+str_parity[i] for i in range(nrows)]
    asdf.insert(1,'Term',spect_term)
    return

def include_NIST(asdf,nistdf):
    ''' Include NIST energy values in term/level dataframe. 
        Missing values are flagged with values < 0. '''
    enist=[]
    for (i,j) in zip(asdf['Configuration'],asdf['Term']):
        try:
            ener=nistdf.loc[(nistdf['Configuration']==i)&(nistdf['Term']==j)]['Level(Ry)'].tolist()[0]
        except:
            ener=-999
        enist.append(ener)
    asdf['NIST']=enist
    return

def write_dstg1():
    ''' Write ICFT code dstg1 input file.
    
        STG1 options:
    
         ISMITN    : =1 pseudostate calculation is being carried out
         MAXLA     : maximum value of total angular momentum for the target terms
         MAXLT     : maximum value of total angular momentum of N+1-electron system
         MAXC      : maximum number of the N+1-electron continuum basis orbitals
                     per angular momentum
         MAXE      : maximum scattering energy in Rydbergs (in general it should
                     be half of the maximum eigen-energy of the continuum orbitals
                     basis orbitals MAXC)
         NMIN      : minimum n value for the pseudostates
         NMAX      : maximum n value for the pseudostates
         LMIN      : minimum l value for the pseudostates
         LMAX      : maximum l value for the pseudostates

    '''
    fname='dstg1'
    header=oompaloompa_header(fname)
    stg1A=namelist_STG1A()
    stg1B=namelist_STG1B()
    with open(fname,'w+') as dstg1:
        dstg1.writelines([header,'\n'])
        dstg1.writelines([stg1A,'\n'])
        dstg1.writelines([stg1B,'\n'])
    print('Write '+fname+' OK')
    return

def namelist_STG1A():
    pseudo=datainp['psorbitals']['pseudo']
    stg1A=['&STG1A']
    if pseudo==True: stg1A.append('ISMITN=1')
    stg1A.append('&END')
    return ' '.join(stg1A)

def namelist_STG1B():
    maxla=SLP_maxangmomentum()
    maxlt=datainp['jmax_ex']
    maxc=datainp['maxc']
    maxe=datainp['maxe']
    pseudo=datainp['psorbitals']['pseudo']
    orbs=datainp['orbs']['qnumbs']
    ipsorb=datainp['psorbitals']['ipsorb']
    norb=datainp['orbs']['norb']
    nmin=orbs[ipsorb][0]
    lmin=orbs[ipsorb][1]
    nmax=orbs[norb-1][0]
    lmax=orbs[norb-1][1]
    stg1B=["&STG1B","MAXLA="+str(maxla),"MAXLT="+str(maxlt),"MAXC="+str(maxc),"MAXE="+str(maxe)]
    if pseudo==True: stg1B+=["NMIN="+str(nmin),"NMAX="+str(nmax),"LMIN="+str(lmin),"LMAX="+str(lmax)]
    stg1B.append("&END")
    return ' '.join(stg1B)

def SLP_maxangmomentum():
    ''' Gives the maximum angular momentum of the SLP configurations yielded. '''
    lq=[i for i in AS['Terms']['L']]
    return max(lq)

def write_dstg2():
    ''' Write ICFT code dstg2 input file.
    
        STG2 options:

         ISORT     : =1 considers all N-electron terms of the same SLP symmetry
                     together, regardless of the order in which they are input.
         NPROCSTG1 : tells the code how many RK from STG1 to read
         MAXORB    : number of orbitals used to define the target configuration
         NELC      : number of target electrons
         NAST      : number of terms to be included in the CC expansion of
                     the target
         INAST     : =0 generates all possible (N+1)-electron SLP symmetries
                     according to MINLT,MAXLT,MINST,MAXST
         MINLT     : minimum value of L of of N+1-electron system
         MAXLT     : maximum value of L of of N+1-electron system
         MINST     : minimum value of 2S+1
         MAXST     : maximum value of 2S+1
    '''
    fname='dstg2'
    header=oompaloompa_header(fname)
    stg2A=namelist_STG2A()
    stg2B=namelist_STG2B()
    ncfg=datainp['cfgs']['ncfg']
    np1cfg=len(datainp['cfgs']['np1_cfgs'])
    with open(fname,'w+') as dstg2:
        dstg2.writelines([header,'\n'])
        dstg2.writelines([stg2A,'\n'])
        dstg2.writelines([stg2B,'\n'])
        write_orbital_qnumbs(dstg2)
        # N electron system:
        dstg2.writelines([str(ncfg),'\n'])
        write_extrema_eocc('N',dstg2)
        write_esystem('N',dstg2,fname)
        write_SLP(dstg2)
        # N+1 electron system:
        dstg2.writelines([str(np1cfg),'\n'])
        write_extrema_eocc('N+1',dstg2)
        write_esystem('N+1',dstg2,fname)
    print('Write '+fname+' OK')
    return

def namelist_STG2A():
    nproc=datainp['nproc']
    stg2A=['&STG2A','ISORT=1','NPROCSTG1='+str(nproc),'&END']
    return ' '.join(stg2A)

def namelist_STG2B():
    norb=datainp['orbs']['norb']
    nelec=datainp['element']['nelec']
    nterms=len(AS['Terms'])
    maxlt=datainp['jmax_ex']
    smax=int((max(AS['Terms']['S'])-1)/2)
    minst,maxst=2*smax,2*(smax+1)
    stg2B=['&STG2B','MAXORB='+str(norb),'NELC='+str(nelec),
           'NAST='+str(nterms),'INAST=0',
           'MINLT=0','MAXLT='+str(maxlt),'MINST='+str(minst),'MAXST='+str(maxst),'&END']
    return ' '.join(stg2B)

def write_extrema_eocc(key,file):
    oneblank=' '
    if key=='N': eocc=datainp['cfgs']['ne_cfgs']
    if key=='N+1': eocc=datainp['cfgs']['np1_cfgs']
    ncfg=len(eocc)
    norb=len(eocc[0])
    eocc_min=' '
    eocc_max=' '
    for i in range(norb):
        iorb=[eocc[j][i] for j in range(ncfg)]
        eocc_min+=str(min(iorb))+oneblank
        eocc_max+=str(max(iorb))+oneblank
    file.writelines([eocc_min,'\n'])
    file.writelines([eocc_max,'\n'])
    return

def write_SLP(file):
    oneblank=' '
    as_terms=AS['Terms']
    SLP_df=as_terms[['S','L','P']].copy()
    nterms=len(SLP_df)
    for i in range(nterms):
        iSLP_df=SLP_df.loc[i].tolist()
        slp=' '
        for i in iSLP_df:
            slp+=str(i)+oneblank
        file.writelines([slp,'\n'])
    return

def write_dstg3():
    ''' Write ICFT code dstg3 input file. 
    
        STG3 options:
        
    '''
    fname='dstg3'
    header=oompaloompa_header(fname)
    prediag=namelist_prediag()
    stg3A=namelist_STG3A()
    stg3B=namelist_STG3B()
    matrixdat=namelist_MATRIXDAT()
    with open(fname,'w+') as dstg3:
        dstg3.writelines([header,'\n'])
        dstg3.writelines([prediag,'\n'])
        dstg3.writelines([stg3A,'\n'])
        dstg3.writelines([stg3B,'\n'])
        dstg3.writelines([matrixdat,'\n'])
        write_shiftenergies(dstg3)
    print('Write '+fname+' OK')
    return

def namelist_prediag():
    jmax_ex=datainp['jmax_ex']
    npw=4*(jmax_ex+1)
    prediag=["&prediag","npw_per_subworld="+str(npw),"&END"]
    return ' '.join(prediag)

def namelist_STG3A():
    stg3A=["&STG3A","ISORT=1","&END"]
    return ' '.join(stg3A)

def namelist_STG3B():
    nshift=datainp['nshift']
    stg3B=["&STG3B","INAST=0","NAST="+str(nshift),"&END"]
    return ' '.join(stg3B)

def namelist_MATRIXDAT():
    nproc=float(datainp['nproc'])
    sqrt_proc=int(np.sqrt(nproc))
    if sqrt_proc**2!=nproc: raise ValueError('The number of processor in input must be square.')
    matrixdat=["&MATRIXDAT","NPROW="+str(sqrt_proc),"NPCOL="+str(sqrt_proc),"&END"]
    return ' '.join(matrixdat)

def write_shiftenergies(file):
    oneblank=' '
    nshift=datainp['nshift']
    as_terms=AS['Terms']
    if nshift!=0:
        shift_df=as_terms.copy()
        ldrop=shift_df.index[shift_df.loc[:]['NIST']<0]
        shift_df.drop(ldrop,axis=0,inplace=True)
        eshift=shift_df['NIST'][:nshift].tolist()
        jj=oneblank
        for i in eshift: 
            jj+=str(i)+2*oneblank
        file.writelines([jj,'\n'])
    return

def write_dstgf(num):
    ''' Write ICFT code dstg3 input file. 
    
        STGF options:
        
         IMESH     : =1 constant spacing in energy dE
                     =2 constant spacing in effective quantum number dn
                     =3 an arbitrary set of user-supplied energies
         IQDT      : =0 standard pstgf operation
                     =1 Multi-channel Quantum Defect Theory operation
                     =2 MQDT operation via unphysical K matrices
         PERT      : 'NO' neglect long-range coupling potentials
                     'YES' include long-range coupling potentials
         IPRINT    :
         MAXLT     :
         LRGLAM    : <0 no top up included
                     >=0 maximum L value for LS, and 2J for IC, partial wave
                     sum will be topped-up (from LRGLAM+1 to infinity)
         IRD0      :
         NOMWRT    : 

    '''
    fname='dstgf_'+str(num)
    stgf=namelist_stgf()
    mesh=namelist_mesh(num)
    with open(fname,'w+') as dstg3:
        dstg3.writelines([stgf,'\n'])
        dstg3.writelines([mesh,'\n'])
    print('Write '+fname+' OK')
    return

def namelist_stgf():
    maxlt=datainp['jmax_ex']
    lrglam=maxlt
    nomwrt=50
    stgf=["&STGF","IMESH=1","IQDT=0","PERT='YES'","IPRINT=-2",
          "MAXLT="+str(maxlt),"LRGLAM="+str(lrglam),"IRD0=101","NOMWRT="+str(nomwrt),"&END"]
    return ' '.join(stgf)

def namelist_mesh(num):
    mxe,e0,eincr=determine_energy_params(num)
    mesh=["&MESH1","MXE="+str(mxe),"E0="+str(e0),"EINCR="+str(eincr),"&END"]
    return ' '.join(mesh)

def determine_energy_params(num):
    efex=NIST['Terms']['Level(Ry)'][1]
    grid_ener=datainp['grid_ener']
    if (grid_ener[0]!=efex): grid_ener.insert(0,efex)
    mxe=datainp['grid_pts'][num-1]
    eincr=(grid_ener[num]-grid_ener[num-1])/mxe
    e0=grid_ener[num-1]-eincr
#     print(eincr,e0,e0+eincr*mxe)
    return mxe,e0,eincr

load_input()
write_das()
run_AS()
load_NIST()
load_AS()
write_dstg1()
write_dstg2()
write_dstg3()
write_dstgf(1)
write_dstgf(2)

# AM 7/1/2020
