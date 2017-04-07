# Dependency test for CS270 HW5

import traceback

installed_numpy = False
installed_scipy = False
installed_sklearn = False

try:
    import numpy
    installed_numpy = True
except ImportError:
    print ''
    print 'Unable to import numpy:'
    traceback.print_exc()
except:
    print 'Non-ImportError when importing numpy'
    traceback.print_exc()
else:
    print 'Successfully imported numpy'

try:
    import scipy
    installed_scipy = True
except ImportError:
    print ''
    print 'Unable to import scipy:'
    traceback.print_exc()
except:
    print 'Non-ImportError when importing scipy'
    traceback.print_exc()
else:
    print 'Successfully imported scipy'

try:
    import sklearn
    installed_sklearn = True
except ImportError:
    print ''
    print 'Unable to import sklearn:'
    traceback.print_exc()
except:
    print 'Non-ImportError when importing sklearn'
    traceback.print_exc()
else:
    print 'Successfully imported sklearn'

print ''
print ''

if installed_numpy and installed_scipy and installed_sklearn:
    print 'All dependencies installed and importable'
else:
    print 'One or more dependencies could not be imported. See messages above'