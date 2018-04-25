from scipy.ndimage.interpolation import shift
from skimage.io import imsave
from tqdm import tqdm 

out = 'chars/bg/bg_%s.jpg'
if not os.path.exists( 'chars/bg' ):
    os.makedirs( 'chars/bg')

files = [
    os.path.join( 'chars', letter, fn )
    for letter in os.listdir( 'chars' )
    for fn in os.listdir( os.path.join( 'chars', letter ))
]

files = np.random.choice(files, size=1000)


images = []
for i,fn in tqdm( enumerate( files ), total=len(files)):
    
    save_name = out%i
    img = imread( fn )
    h,w = img.shape
    amount_h = np.random.randint( h//2, h ) * np.random.choice([-1,1])
    amount_w = np.random.randint( w//2, h ) * np.random.choice([-1,1])

    bg = shift( img, (amount_h, amount_w), cval=255 )
    imsave( save_name, bg )
    
