# Advanced Lane Finding

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./writeupimages/undistort1.png "Undistorted Chess"
[image2]: ./writeupimages/undistort2.png "Road Undistorted"
[image3]: ./writeupimages/thresholded.png "Binary Example"
[image4]: ./writeupimages/warped.png "Perspective Transform"
[image5]: ./writeupimages/histogram.png "Histogram"
[image6]: ./writeupimages/slidingwindow.png "Sliding Window"
[image7]: ./writeupimages/seachprior.png "Search Prior"
[image8]: ./writeupimages/inversetransform.png "Output"
[video1]: ./project_video_output.mp4 "Output Video"

## Camera Calibration

The given chess board has 9 corners in a row and 6 corners in a column. Using the given chess board images we extract the camera matrix and distortion coefficients using the function `calibrateCamera()` as shown below. 

```python

#Creating arrays to store object points and image. points
objpoints = [] #3d points in real world
imgpoints = [] #2d points in an image

#Preparing Object points
objp = np.zeros((9*6,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

images = glob.glob("camera_cal/calibration*.jpg")

for image in images: 
    img = mpimg.imread(image)
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(gray,(9,6),None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None

```

Now using 'mtx' and 'dist' (camera matrix and distortion coefficients) we can undistort the image by calling the function `undistort()`

```python
undistorted = cv2.undistort(img, mtx, dist, None, mtx)
```

![alt text][image1]
![alt text][image2]


## Color transform and Gradients

I used a combination of color and gradient thresholds to generate a binary image. The function `colorandgradient2()` contains all the steps for thresholding.

```python
def colorandgradient2(img, s_thresh=(40,255), sx_thresh=(10,200)):
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    l_channel = hls[:,:,1]
    
    
    # R & G thresholds so that yellow lanes are detected well.
    R = img[:,:,0]
    G = img[:,:,1]
    color_threshold = 180
    r_g = np.zeros_like(R)
    r_g[(R > color_threshold) & (G > color_threshold)] = 1
    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    #Sobelx
    sobelx = cv2.Sobel(gray,cv2.CV_64F,1,0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    #Threshold x gradient
    sx_binary = np.zeros_like(scaled_sobel)
    sx_binary[(scaled_sobel >= sx_thresh[0])&(scaled_sobel <= sx_thresh[1])] = 1
    
    #Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0])&(s_channel <= s_thresh[1])] = 1
    
    # We put a threshold on the L channel to avoid pixels which have shadows and as a result darker.
    l_thresh = (180, 255)
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel > l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sx_binary)
    combined_binary[(s_binary == 1) & (sx_binary == 1) | (l_binary == 1) | (r_g == 1) ] = 1
    return combined_binary
```

![alt text][image3]


## Persepective Transform

To perform a perspective transform, I manually selected the vertices of the source image and then warped them to the different points in the destination image. This is achieved using the functions `getPerspectiveTransform()` and `warpPerspective()`.

```python
# Vertices extracted manually for performing a perspective transform
bottom_left = [270,720] 
bottom_right = [1160, 720] 
top_left = [580, 470]
top_right = [730, 470]


source = np.float32([bottom_left,bottom_right,top_right,top_left])


pts = np.array([bottom_left,bottom_right,top_right,top_left], np.int32)
pts = pts.reshape((-1,1,2))
copy = img.copy()
cv2.polylines(copy,[pts],True,(255,0,0), thickness=3)

# Destination points are chosen such that straight lanes appear more or less parallel in the transformed image.
bottom_left = [320,720]
bottom_right = [920, 720]
top_left = [320, 1]
top_right = [920, 1]


dst = np.float32([bottom_left,bottom_right,top_right,top_left])

M = cv2.getPerspectiveTransform(source, dst)

M_inv = cv2.getPerspectiveTransform(dst, source)
img_size = (imshape[1], imshape[0])

warped = cv2.warpPerspective(uthresholded, M, img_size , flags=cv2.INTER_LINEAR)

    
f, (ax1, ax2) = plt.subplots(1, 2,  figsize=(24, 9))
f.tight_layout()
#ax1.imshow(copy)
ax1.imshow(roi_select, cmap='gray')
ax1.set_title('Thresholded Image', fontsize=50)
ax2.imshow(warped, cmap='gray')
ax2.set_title('Warped Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
```


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 470      | 320, 1        | 
| 220, 720      | 320, 720      |
| 1110, 720     | 920, 720      |
| 720, 470      | 920, 1        |


![alt text][image4]


## Detecting Lane Pixels

### Histogram

We get the likely position of lanes using the histogram function

```python
#Defining a function to return a histogram of image binary activations
def hist(img):
    #Selecting only bottom half of the image
    bottom_half = img[img.shape[0]//2:,:]
    #sum across image pixels vertically
    histogram = np.sum(bottom_half, axis=0)
    
    return histogram
```

![alt text][image5]


### Sliding Window Search

If we don't have any lane positions then we do sliding window search initially, starting with the likely lane positions calculated from the histgram.  For this, I'm using 10 windows of width 100 pixels. Once we detect the lane pixels, a polynomial is fit and the lane lines are drawn. The following is the code for `slidingWindow()` function.


```python
def slidingWindow(warped):
    
    himg = warped/255
    #create histogram
    histogram = hist(himg)
    
    #To find the starting point of the left and right lanes, we find the peak of the left and right halves of the histogram
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:])+midpoint
    
    # Setting up windows and window hyperparameters
    #number of sliding windows
    nwindows = 9
    #width of the windows
    margin = 100
    #minimum number of pixels found to recenter window
    minpix = 50

    #Creating an output image to draw the results
    out_img = np.dstack((warped, warped, warped))

    #height of windows based on nwindows above and image shape
    num_rows = warped.shape[0]
    window_height = np.int(warped.shape[0]//nwindows)

    #Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    #empty lists to recieve left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        #Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
    
        #Draw the windows on visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    
        # Identify the nonzero pixels in x and y within the window 
        #for left
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        #for right
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
    
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
    
        # If found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    # Concatenate the arrays of indices 
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
    # Avoids an error 
        pass
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    #Fitting a polynomial to all the relevant pixels

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fitx, right_fitx, ploty
```

![alt text][image6]


### Search from prior lane positions

As we know that consecutive frames of the video will have lane lines in roughly similar positions, here we search around a margin of 50 pixels of the previously detected lane lines. The following code is for `SearchPrior()` function.

```python
def searchPrior(binary_warped, left_fitx, right_fitx):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    
    return left_fitx, right_fitx, ploty
```

![alt text][image7]


## Determining curvature radius and vehicle position

***Radius of curvature***
The radius of curvature is calculated according to the method described in the classroom. Since we perform the polynomial fit in pixels we need to calulate radius in real world meters.

***Offset from the center***
The mean of the lane pixels closest to the car gives us the center of the lane. The center of the image gives us the position of the car. The difference between them is the offset from the center.

Code for calculating radius and offset:

```python

def radiusOfCurvature(values):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(ploty*ym_per_pix, values*xm_per_pix, 2)
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
    return curverad

# Obtaining the radius
left_curverad = radiusOfCurvature(left_fitx)
right_curverad = radiusOfCurvature(right_fitx)
rad = (left_curverad+right_curverad)/2

# obtaining the offset from the center
lane_center = (right_fitx[719] + left_fitx[719])/2
xm_per_pix = 3.7/700 # meters per pixel in x dimension
offset_pixels = abs(img_size[0]/2 - lane_center)
offset_mtrs = xm_per_pix*offset_pixels

curvature = "Radius of curvature: %.2f m" % rad
offset = "Center offset: %.2f m" % offset_mtrs

print("Radius of curvature: %.2f m" % rad)
print("Center offset: %.2f m" % offset_mtrs)
```

## Plotting the result on the image

Here we paint the area between lanes and display the output image. The following is the code:

```python
out_img = np.dstack((warped, warped, warped))*255

ploty = np.linspace(0, num_rows-1, num_rows)

left_line_window = np.array(np.transpose(np.vstack([left_fitx, ploty])))

right_line_window = np.array(np.flipud(np.transpose(np.vstack([right_fitx, ploty]))))

line_points = np.vstack((left_line_window, right_line_window))

cv2.fillPoly(out_img, np.int_([line_points]), [0,255, 0])

unwarped = cv2.warpPerspective(out_img, M_inv, img_size , flags=cv2.INTER_LINEAR)

result = cv2.addWeighted(uimg, 1, unwarped, 0.3, 0)

cv2.putText(result, curvature , (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)
cv2.putText(result, offset, (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), thickness=2)

plt.imshow(result)
```

![alt text][image8]

---

# Pipeline (video)

Here's a [link to my video result](./project_video_output.mp4)

---

# Discussion

 - First I had to experiment a lot with the color and gradient thresholding.
 
 - The pipeline works well for the project video but it struggles with the challenge video.
 
 - Tuning the color and gradient thresholds and decreasing the size of region of interest could help with the challenging videos.
