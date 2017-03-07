from collections import defaultdict
import time

#########################  Problem 1 and 2 code below #####################

class Constraint:
    """A constraint of a CSP.  Members include
     - name: a string for debugging
     - domain, a list of variables on which the constraint acts
     - predicate, a boolean function with the same arity as the domain.
     """
    def __init__(self,name,domain,pred):
        self.name = name
        self.domain = domain
        self.predicate = pred
        
    def isSatisfied(self,vars):
        """Given a dictionary of variables, evaluates the predicate.
        If a variable in the domain isn't present, raises a KeyError."""
        args = [vars[v] for v in self.domain]
        return self.predicate(*args)

class CSP:
    """Defines a constraint satisfaction problem.  Contains 4 members:
    - variables: a list of variables
    - domains: a dictionary mapping variables to domains
    - constraints: a list of Constraints.
    - incidentConstraints: a dict mapping each variable to a list of
      constraints acting on it.
    """
    
    def __init__(self,variables=[],domains=[]):
        """Input: a list of variables and a list of domains.

        Note: The variable names must be unique, otherwise undefined behavior
        will result.
        """
        self.variables = variables[:]
        self.domains = dict(zip(variables,domains))
        self.constraints = []
        self.incidentConstraints = dict((v,[]) for v in variables)
        
    def addVariable(self,var,domain):
        """Adds a new variable with a given domain.  var must not already
        be present in the CSP."""
        if var in self.domains:
            raise ValueError("Variable with name "+val+" already exists in CSP")
        self.variables.append(var)
        self.domains[var] = domain
        self.incidentConstraints[var] = []

    def addConstraint(self,varlist,pred,name=None):
        """Adds a constraint with the domain varlist, the predicate pred,
        and optionally a name for printing."""
        if name==None:
            name = "c("+",".join(str(v) for v in varlist)+")"
        self.constraints.append(Constraint(name,varlist,pred))
        for v in varlist:
            self.incidentConstraints[v].append(self.constraints[-1])

    def addUnaryConstraint(self,var,pred,name=None):
        """Adds a unary constraint with the argument var, the predicate pred,
        and optionally a name for printing."""
        self.addConstraint((var,),pred,name)

    def addBinaryConstraint(self,var1,var2,pred,name=None):
        """Adds a unary constraint with the arguments (var1,var2), the
        predicate pred, and optionally a name for printing."""
        self.addConstraint((var1,var2),pred,name)

    def fixValue(self,var,value,name=None):
        """Adds a constraint that states var = value."""
        if name==None:
            name = str(var)+'='+str(value)
        self.addUnaryConstraint(var,lambda x:x==value,name)

    def nAryConstraints(self,n,var=None):
        """Returns a list of all n-ary constraints in the CSP if var==None,
        or if var is given, returns a list of all n-ary constraints involving
        var."""
        if var==None:
            return [c for c in self.constraints if len(c.domain)==n]
        else:
            return [c for c in self.incidentConstraints[var] if len(c.domain)==n]

    def incident(self,*vars):
        """incident(var1,...,varn) will return a list of constraints
        that involve all of var1 to varn."""
        if len(vars)==0: return self.constraints
        res = set(self.incidentConstraints[vars[0]])
        for v in vars[1:]:
            res &= set(self.incidentConstraints[v])
        return [c for c in res]

    def isConstraintSatisfied(self,c,partialAssignment):
        """Checks if the partial assignment satisfies the constraint c.
        If the partial assignment doesn't cover the domain, this returns
        None. """
        try:
            res = c.isSatisfied(partialAssignment)
            return res
        except KeyError:
            return None

    def isValid(self,partialAssignment,*vars):
        """Checks if the assigned variables in a partial assignment
        are mutually compatible.  Only checks those constraints
        involving assigned variables, and ignores any constraints involving
        unassigned ones.

        If no extra arguments are given, checks all constraints relating
        assigned variables.
        
        If extra arguments var1,...,vark are given, this only checks
        constraints that are incident to those given variables."""
        for c in self.incident(*vars):
            #all entries in partialAssignment must be in the domain of c
            #for this to be checked
            if self.isConstraintSatisfied(c,partialAssignment)==False:
                return False
        return True

def streetCSP():
    """Returns a CSP corresponding to the street puzzle covered in class."""
    nationalityVars = ['N1','N2','N3','N4','N5']
    colorVars = ['C1','C2','C3','C4','C5']
    drinkVars = ['D1','D2','D3','D4','D5']
    jobVars = ['J1','J2','J3','J4','J5']
    animalVars = ['A1','A2','A3','A4','A5']
    nationalities = ['E','S','J','I','N']
    colors = ['R','G','W','Y','B']
    drinks = ['T','C','M','F','W']
    jobs = ['P','S','Di','V','Do']
    animals = ['D','S','F','H','Z']
               
    csp = CSP(nationalityVars+colorVars+drinkVars+jobVars+animalVars,
              [nationalities]*5+[colors]*5+[drinks]*5+[jobs]*5+[animals]*5)
    
    #TODO: fill me in.  Slide 18 is filled in for you.  Don't forget to enforce
    #that all nationalities, colors, drinks, jobs, and animals are distinct!
    
    #Englishman lives in the red house
    for Ni,Ci in zip(nationalityVars,colorVars):
        csp.addBinaryConstraint(Ni,Ci,lambda x,y:(x=='E')==(y=='R'),'Englishman lives in the red house')
    #The Spaniard has a Dog
    for Ni,Ai in zip(nationalityVars,animalVars):
        csp.addBinaryConstraint(Ni,Ai,lambda x,y:(x=='S')==(y=='D'),'The Spaniard has a Dog')
    #Japanese is a painter
    for Ni,Ji in zip(nationalityVars,jobVars):
        csp.addBinaryConstraint(Ni,Ji,lambda x,y:(x=='J')==(y=='P'),'Japanese is a painter')
    #The Italian drinks Tea
    for Ni,Di in zip(nationalityVars,drinkVars):
        csp.addBinaryConstraint(Ni,Di,lambda x,y:(x=='I')==(y=='T'),'The Italian drinks Tea')
    #Norwegian lives in first house
    csp.fixValue('N1','N','Norwegian lives in the first house')
    #The owner of the Green house drinks Coffee
    for Ci,Di in zip(colorVars,drinkVars):
        csp.addBinaryConstraint(Ci,Di,lambda x,y:(x=='G')==(y=='C'),'The owner of the Green house drinks Coffee')
    #green house is to the right of the white house
    for Ci,Cn in zip(colorVars[:-1],colorVars[1:]):
        csp.addBinaryConstraint(Ci,Cn,lambda x,y:(x=='W')==(y=='G'),'Green house is to the right of the white house')
    csp.addUnaryConstraint('C5',lambda x:x!='W','Green house is to the right of the white house')
    csp.addUnaryConstraint('C1',lambda x:x!='G','Green house is to the right of the white house')
    #The Sculptor breeds Snails
    for Ji,Ai in zip(jobVars,animalVars):
        csp.addBinaryConstraint(Ji,Ai,lambda x,y:(x=='S')==(y=='S'),'The Sculptor breeds Snails')
    #The Diplomat lives in the Yellow house
    for Ji,Ci in zip(jobVars,colorVars):
        csp.addBinaryConstraint(Ji,Ci,lambda x,y:(x=='Di')==(y=='Y'),'The Diplomat lives in the Yellow house')
    #The owner of the middle house drinks Milk
    csp.fixValue('D3','M','The owner of the middle house drinks Milk')
    #The Norwegian lives next door to the Blue house
    
    for Ai,Ji,An in zip(nationalityVars[0:3],colorVars[1:4],nationalityVars[2:5]):
        t=[Ai,Ji,An]
        csp.addConstraint(t,lambda a,b,c:(a=='N')==(b=='B') or (b=='B')==(c=='N'),'horse is next to house of diplomat')
    for Ai,Ji,An in zip(animalVars[0:3],jobVars[1:4],animalVars[2:5]):
        t=[Ai,Ji,An]
        csp.addConstraint(t,lambda a,b,c:(a=='F')==(b=='Do') or (b=='Do')==(c=='F'),'horse is next to house of diplomat')
    for Ai,Ji,An in zip(animalVars[0:3],jobVars[1:4],animalVars[2:5]):
        t=[Ai,Ji,An]
        csp.addConstraint(t,lambda a,b,c:(a=='H')==(b=='Di') or (b=='Di')==(c=='H'),'horse is next to house of diplomat')

    #The Violinist drinks Fruit juice
    for Ji,Di in zip(jobVars,drinkVars):
        csp.addBinaryConstraint(Ji,Di,lambda x,y:(x=='V')==(y=='F'),'The Violinist drinks Fruit juice')

    #Enforcing each are distinct
    #csp.addConstraint(nationalityVars, lambda a,b,c,d,e:(a!=b!=c!=d!=e))
    csp.addConstraint(nationalityVars, lambda a,b,c,d,e:(a!=b)==(a!=c)==(a!=d)==(a!=e)==(b!=c)==(b!=d)==(b!=e)==(c!=d)==(c!=e)==(d!=e))
    csp.addConstraint(colorVars, lambda a,b,c,d,e:(a!=b)==(a!=c)==(a!=d)==(a!=e)==(b!=c)==(b!=d)==(b!=e)==(c!=d)==(c!=e)==(d!=e))
    csp.addConstraint(drinkVars, lambda a,b,c,d,e:(a!=b)==(a!=c)==(a!=d)==(a!=e)==(b!=c)==(b!=d)==(b!=e)==(c!=d)==(c!=e)==(d!=e))
    csp.addConstraint(jobVars, lambda a,b,c,d,e:(a!=b)==(a!=c)==(a!=d)==(a!=e)==(b!=c)==(b!=d)==(b!=e)==(c!=d)==(c!=e)==(d!=e))
    csp.addConstraint(animalVars, lambda a,b,c,d,e:(a!=b)==(a!=c)==(a!=d)==(a!=e)==(b!=c)==(b!=d)==(b!=e)==(c!=d)==(c!=e)==(d!=e))


    print "CSP has",len(csp.constraints),"constraints"

    #TODO:
    return csp

def p1():
    csp = streetCSP()
    solution = dict([('A1', 'F'), ('A2', 'H'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'W'), ('D2', 'T'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])
    invalid1 = dict([('A1', 'F'), ('A2', 'H'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'T'), ('D2', 'W'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])
    invalid2 = dict([('A1', 'F'), ('A2', 'F'), ('A3', 'S'), ('A4', 'D'), ('A5', 'Z'),
                     ('C1', 'Y'), ('C2', 'B'), ('C3', 'R'), ('C4', 'W'), ('C5', 'G'),
                     ('D1', 'W'), ('D2', 'T'), ('D3', 'M'), ('D4', 'F'), ('D5', 'C'),
                     ('J1', 'Di'), ('J2', 'Do'), ('J3', 'S'), ('J4', 'V'), ('J5', 'P'),
                     ('N1', 'N'), ('N2', 'I'), ('N3', 'E'), ('N4', 'S'), ('N5', 'J')])
    print "Valid assignment valid?",csp.isValid(solution)
    print "Invalid assignment valid?",csp.isValid(invalid1)
    print "Invalid assignment valid?",csp.isValid(invalid2)
    
    #you may wish to check the solver once you've solved problem 2
    #solver = CSPBacktrackingSolver(csp)
    #res = solver.solve()
    #print "Result:",sorted(res.items())

############################  Problem 2 code below #######################
count = 0

class CSPBacktrackingSolver:
    """ A CSP solver that uses backtracking.
    A state is a partial assignment dictionary {var1:value1,...,vark:valuek}.
    Also contains a member oneRings that is a dict mapping each variable to
    all variables that share a constraint.
    """
    def __init__(self,csp,doForwardChecking=True,doConstraintPropagation=False):
        self.csp = csp
        self.doForwardChecking = doForwardChecking
        self.doConstraintPropagation = doConstraintPropagation
        #compute 1-rings
        self.oneRings = dict((v,set()) for v in csp.variables)
        for c in csp.constraints:
            cdomain = set(c.domain)
            for v in c.domain:
                self.oneRings[v] |= cdomain
        for v in csp.variables:
            if v in self.oneRings[v]:
                self.oneRings[v].remove(v)

    def solve(self):
        """Solves the CSP, returning an assignment if solved, or False if
        failed."""
        domains = self.initialDomains()
        return self.search({},domains)

    def search(self,partialAssignment,domains):
        global count

        """Runs recursive backtracking search."""
        if len(partialAssignment)==len(self.csp.variables):
            return partialAssignment
        if self.doConstraintPropagation:
            domains = self.constraintPropagation(partialAssignment,domains)
            #contradiction detected
            if any(len(d)==0 for (v,d) in domains.iteritems()):
                return False
        indent = " "*len(partialAssignment)
        X = self.pickVariable(partialAssignment,domains)
        values = self.orderValues(partialAssignment,domains,X)
        for v in values:
            partialAssignment[X] = v
            count = count + 1
            if self.doForwardChecking:
                print indent+"Trying",X,"=",v
                #do forward checking
                newDomains = self.forwardChecking(partialAssignment,X,domains)
                if any(len(d)==0 for (v,d) in newDomains.iteritems()):
                    #contradiction, go on to next value
                    emptyvars = [v for (v,d) in newDomains.iteritems() if len(d)==0]
                    print indent+" Forward checking found contradiction on",emptyvars[0]
                    continue
                #recursive call
                res = self.search(partialAssignment,newDomains)
                if res!=False: return res
            else:
                #check whether the assignment X=v is valid
                if self.csp.isValid(partialAssignment,X):
                    print indent+"Trying",X,"=",v
                    #recursive call
                    res = self.search(partialAssignment,domains)
                    if res!=False: return res
        #remove the partial assignment to X, backtrack
        del partialAssignment[X]
        return False
        
    def initialDomains(self):
        """Does the basic step of checking all unary constraints"""
        domains = dict()
        for v,domain in self.csp.domains.iteritems():
            #save only valid constraints
            vconstraints = self.csp.nAryConstraints(1,v)
            dvalid = [val for val in domain if all(c.predicate(val) for c in vconstraints)]
            domains[v] = dvalid
        return domains

    def pickVariable(self,partialAssignment,domains):
        """Return an unassigned variable to assign next"""
        #TODO (Problem 2): implement heuristics
        return [v for v,domain in domains.iteritems() if v not in partialAssignment][0]

    def orderValues(self,partialAssignment,domains,var):
        """Return an ordering on the domain domains[var]"""
        #TODO (Bonus): implement heuristics.  Currently doesn't do anything
        return domains[var]

    def constraintPropagation(self,partialAssignment,domains):
        """domains is a dict mapping vars to valid values.
        Return a copy of domains but with all invalid values removed."""
        #TODO (Bonus): implement AC3. Currently doesn't do anything
        return domains

    def forwardChecking(self,partialAssignment,var,domains):
        """domains is a dict mapping vars to valid values.  var has just been
        assigned.
        Return a copy of domains but with all invalid values removed"""
        resdomain = dict()
        #do a shallow copy for all unaffected domains, this saves time
        for v,domain in domains.iteritems():
            resdomain[v] = domain
        resdomain[var] = [partialAssignment[var]]
        print dict(resdomain)
        
        #TODO: uncomment this line to perform forward checking
        #return resdomain
        
        #TODO: perform forward checking on binary constraints
        #NOTE: be sure not to modify the resdomains directly, but to create
        #      new lists 
        for c in self.csp.incidentConstraints[var]:
            #If the domain has size k and exactly k-1 entries are filled, then
            #do forward checking.  If so, 'unassigned' will contain the name of
            #the unassigned variable.
            kassigned = 0
            unassigned = None
            for v in c.domain:
                if v in partialAssignment:
                    kassigned += 1
                else:
                    unassigned = v
            if kassigned+1 == len(c.domain):
                print "Forward checking",unassigned
                validvalues = []

                partialCopy = dict()
                for a,b in partialAssignment.iteritems():
                    partialCopy[a] = b
                
                #TODO (Problem 2): check whether each values in the domain of unassigned
                #(resdomain[unassigned]) is compatible under c. May want to use
                #self.csp.isConstraintSatisfied(c,assignment).  If compatible,
                #append the value to validvalues

                for value in resdomain[unassigned]:
                    partialCopy.pop(unassigned, None)
                    partialCopy[unassigned] = value
                    if self.csp.isConstraintSatisfied(c,partialCopy):
                        validvalues.append(value)

                #values = self.csp.isConstraintSatisfied(c,[unassigned,resdomain[unassigned]])
                #if values != None:
                #    validvalues.append(values)
                
                resdomain[unassigned] = validvalues
                if len(validvalues)==0:
                    #print "Domain of",unassigned,"emptied due to",c.name
                    #early terminate, this setting is a contradiction
                    return resdomain
        return resdomain

def nQueensCSP(n):
    """Returns a CSP for an n-queens problem"""
    vars = ['Q'+str(i) for i in range(1,n+1)]
    domain = range(1,n+1)
    csp = CSP(vars,[domain]*len(vars))
    for i in range(1,n+1):
        for j in range(1,i):
            Qi = 'Q'+str(i)
            Qj = 'Q'+str(j)
            ofs = i-j
            #this weird default argument thing is needed for lambda closure
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y: x!=y),Qi+"!="+Qj)
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y,ofs=ofs: x!=(y+ofs)),Qi+"!="+Qj+"+"+str(i-j))
            csp.addBinaryConstraint(Qi,Qj,(lambda x,y,ofs=ofs: x!=(y-ofs)),Qi+"!="+Qj+"-"+str(i-j))
    return csp

def p2():
    global count
    start = time.time()
    csp = nQueensCSP(4)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=False)
    res = solver.solve()
    print "Result:",sorted(res.items())
    print count
    end1 = time.time()
    print end1-start
    raw_input()

    #TODO: implement forward checking, change False to True
    csp = nQueensCSP(8)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=False)
    res = solver.solve()
    print "Result:",sorted(res.items())
    print count
    end2 = time.time()
    print end2-end1
    raw_input()

    csp = nQueensCSP(12)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=False)
    res = solver.solve()
    print "Result:",sorted(res.items())
    print count
    end3 = time.time()
    print end3-end2
    raw_input()

    csp = nQueensCSP(12)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=False)
    res = solver.solve()
    print "Result:",sorted(res.items())
    print count
    end4 = time.time()
    print end4-end3
    raw_input()

    csp = nQueensCSP(20)
    solver = CSPBacktrackingSolver(csp,doForwardChecking=False)
    res = solver.solve()
    print "Result:",sorted(res.items())
    print count
    end5 = time.time()
    print end5-end4

    #TODO: see how high you can crank n!



############################  Problem 4 code below #######################

def marginalize(probabilities,index):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    """
    #TODO (Problem 3): you may hard-code two routines for n=2 and n=3, but there's an
    #elegant solution that uses defaultdict(float)
    if len(probabilities)==4:
        ans = {}
        if index==1:
            ans[(1,)] = probabilities[(1,1)] + probabilities[(1,0)]
            ans[(0,)] = probabilities[0,1] + probabilities[0,0]
            return ans
        elif index==0:
            ans[(1,)] = probabilities[(1,1)] + probabilities[(0,1)]
            ans[(0,)] = probabilities[(1,0)] + probabilities[(0,0)]
            return ans
    elif len(probabilities)==8:
        ans = {}
        if index==2:
            ans[(0,0)] = probabilities[(0,0,1)] + probabilities[(0,0,0)]
            ans[(0,1)] = probabilities[(0,1,1)] + probabilities[(0,1,0)]
            ans[(1,0)] = probabilities[(1,0,1)] + probabilities[(1,0,0)]
            ans[(1,1)] = probabilities[(1,1,1)] + probabilities[(1,1,0)]
            return ans
        elif index==0:
            ans[(0,0)] = probabilities[(1,0,0)] + probabilities[(0,0,0)]
            ans[(0,1)] = probabilities[(1,1,0)] + probabilities[(0,1,0)]
            ans[(1,0)] = probabilities[(1,0,1)] + probabilities[(0,0,1)]
            ans[(1,1)] = probabilities[(1,1,1)] + probabilities[(0,1,1)]
            return ans
        elif index==1:
            ans[(0,0)] = probabilities[(0,1,0)] + probabilities[(0,0,0)]
            ans[(0,1)] = probabilities[(0,1,1)] + probabilities[(0,0,1)]
            ans[(1,0)] = probabilities[(1,1,0)] + probabilities[(1,0,0)]
            ans[(1,1)] = probabilities[(1,1,1)] + probabilities[(1,0,1)]
            return ans

def marginalize_multiple(probabilities,indices):
    """Safely marginalizes multiple indices"""
    pmarg = probabilities
    for index in reversed(sorted(indices)):
        pmarg = marginalize(pmarg,index)
    return pmarg

def condition1(probabilities,index,value):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn | Xi=v).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    - value: the value of v
    """
    #TODO (Problem 3)
    #Compute the denominator by marginalizing over everything but Xi
    num = {}

    if len(probabilities)==4:
        if index==0:
            den = marginalize(probabilities,1)
            print dict(den)
            num[(1,)] = probabilities[(value,1)]/den[(value,)]
            num[(0,)] = probabilities[(value,0)]/den[(value,)]
        elif index==1:
            den = marginalize(probabilities,0)
            print dict(den)
            num[(1,)] = probabilities[(1,value)]/den[(value,)]
            num[(0,)] = probabilities[(0,value)]/den[(value,)]
    elif len(probabilities)==8:
        if index==0:
            den = marginalize(marginalize(probabilities,2),1)
            num[(0,0)] = probabilities[(value,0,0)]/den[(value,)]
            num[(0,1)] = probabilities[(value,0,1)]/den[(value,)]
            num[(1,0)] = probabilities[(value,1,0)]/den[(value,)]
            num[(1,1)] = probabilities[(value,1,1)]/den[(value,)]
        elif index==1:
            den = marginalize(marginalize(probabilities,2),0)
            num[(0,0)] = probabilities[(0,value,0)]/den[(value,)]
            num[(0,1)] = probabilities[(0,value,1)]/den[(value,)]
            num[(1,0)] = probabilities[(1,value,0)]/den[(value,)]
            num[(1,1)] = probabilities[(1,value,1)]/den[(value,)]
        elif index==2:
            den = marginalize(marginalize(probabilities,1),0)
            num[(0,0)] = probabilities[(0,0,value)]/den[(value,)]
            num[(0,1)] = probabilities[(0,1,value)]/den[(value,)]
            num[(1,0)] = probabilities[(1,0,value)]/den[(value,)]
            num[(1,1)] = probabilities[(1,1,value)]/den[(value,)]
    return num

def normalize(probabilities):
    """Given an unnormalized distribution, returns a normalized copy that
    sums to 1."""
    vtotal = sum(probabilities.values())
    return dict((k,v/vtotal) for k,v in probabilities.iteritems())

def condition2(probabilities,index,value):
    """Given a probability distribution P(X1,...,Xi,...,Xn),
    return the distribution P(X1,...,Xi-1,Xi+1,...,Xn | Xi=v).
    - probabilities: a probability table, given as a map from tuples
      of variable assignments to values
    - index: the value of i.
    - value: the value of v
    """
    #TODO (Problem 3)
    #Compute the result by normalizing
    num = {}

    if len(probabilities)==4:
        if index==0:
            num[(1,)] = probabilities[(value,1)]
            num[(0,)] = probabilities[(value,0)]
        elif index==1:
            den = marginalize(probabilities,0)
            print dict(den)
            num[(1,)] = probabilities[(1,value)]
            num[(0,)] = probabilities[(0,value)]
    elif len(probabilities)==8:
        if index==0:
            num[(0,0)] = probabilities[(value,0,0)]
            num[(0,1)] = probabilities[(value,0,1)]
            num[(1,0)] = probabilities[(value,1,0)]
            num[(1,1)] = probabilities[(value,1,1)]
        elif index==1:
            num[(0,0)] = probabilities[(0,value,0)]
            num[(0,1)] = probabilities[(0,value,1)]
            num[(1,0)] = probabilities[(1,value,0)]
            num[(1,1)] = probabilities[(1,value,1)]
        elif index==2:
            num[(0,0)] = probabilities[(0,0,value)]
            num[(0,1)] = probabilities[(0,1,value)]
            num[(1,0)] = probabilities[(1,0,value)]
            num[(1,1)] = probabilities[(1,1,value)]
    return normalize(num)

def p4():
    pAB = {(0,0):0.5,
           (0,1):0.3,
           (1,0):0.1,
           (1,1):0.1}
    pA = marginalize(pAB,1)
    print (pA[(0,)],pA[(1,)]),"should be",(0.8,0.2)

    pABC = {(0,0,0):0.2,
            (0,0,1):0.3,
            (0,1,0):0.06,
            (0,1,1):0.24,
            (1,0,0):0.02,
            (1,0,1):0.08,
            (1,1,0):0.06,
            (1,1,1):0.04}

    print "marginalized p(A,B): ",dict(marginalize(pABC,2))
    pA = marginalize(marginalize(pABC,2),1)
    print dict(pA)
    print (pA[(0,)],pA[(1,)]),"should be",(0.8,0.2)

    pA_B = condition1(pAB,1,1)
    print (pA_B[(0,)],pA_B[(1,)]),"should be",(0.75,0.25)
    pA_B = condition2(pAB,1,1)
    print (pA_B[(0,)],pA_B[(1,)]),"should be",(0.75,0.25)

    pAB_C = condition1(pABC,2,1)
    print "p(A,B|C): ",dict(pAB_C)
    pAB_C = condition2(pABC,2,1)
    print "p(A,B|C): ",dict(pAB_C)

    pA_BC = condition1(condition1(pABC,2,1),1,1)
    print "p(A|B,C): ",dict(pA_BC)
    pA_BC = condition2(condition2(pABC,2,1),1,1)
    print "p(A|BC): ",dict(pA_BC)

if __name__=='__main__':
    print "###### Problem 1 ######"
    p1()
    raw_input()
    print
    print "###### Problem 2 ######"
    p2()
    raw_input()
    print
    print "###### Problem 4 ######"
    p4()
    
