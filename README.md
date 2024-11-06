# Image Compression using Principal Component Analysis (PCA)

Employed PCA to extract out only the most important information from the image by selcting only a few components from the full set of **100** components. This technique can be used for image compression and we can reduce the file-size to **85%** of its original size through **almost lossless** compression. It can also reduce file-size upto **65%** of its original size but through **lossy compression**.

## Author - Shubham Apte, Omkar Nitsure <br>

## Methodology

The input image is first resized in terms of number of pixels to the closest multiple of **10** in both x and y direction. The entire image is then broken down into disjoint patches of (**10, 10**). These pathces are then flattened into a vector of length **100**. This was done independently for all the **3** channels. These vectors were then stored columnwise one after the other to make **3** matrices one for each channel. Then the **covariance matrix** for the matrices were computed after subtracting the **mean**. Then the **eigenvalues and eigenvectors** of the covariance matrix were computed. The eigenvectors were then sorted in the decreasing order of the absolute value of the corresponding eigenvalues. Now based on our choice of the extent of compression, a set of these top eigenvectors are selected for each channel and all vectors in each of the **3** matrices are resolved in the directions of these eigenvectors. The value of projections is stored in **3** new matrices one for each channel. Finally, the vectors corresponding to each of the image patch are reconstructed based on projections on only the selected eigenvectors. The vectors are then reshaped to make the patches and the patches are then placed in their appropriate positions to get the final channel-wise image. These channels are finally combined to get the **compressed image**.

## Results and Conlusions

The results show the true power of PCA, as only the top eigenvector with the highest absolute value of the corresponding eigenvalue out of the total 100 carries almost the entire structural information of the image though there is significant loss of clarity. As we include more and more **components** the clarity of the image increases. We get reasonably good quality image by taking **10** principal components and almost identical image to the original one by taking **20** principal components. The gains in the visually perceptual quality becomes unnoticeable as we keep adding more and more principal components above **20**. In the following example, you can take a look at the changes in image quality as we go on including more and more principal components. <br>

<img src="/Results/Outputs_Comparison.jpg">

## Levels of Image compression in the Flask Application

I have made a Flask application which when given an input image can do **3** different levels of image compression based on the user's choice. The mathematical parameters used in these **3** levels along with their names is given below -> <br>

<ol>
    <li>Lossy Image Compression -> Using <strong>only 1</strong> Principal Component</li>
    <li>Medium Lossy Image Compression -> Using <strong>10</strong> Principal Components</li>
    <li>Almost Loss-less Compression -> Using <strong>20</strong> Principal components</li>
</ol>

You can take a look at one of the example image being compressed by these **3** levels of compression.

<img src="/Results/Levels_of_Compression.jpg">

Image Credit -> <a href="https://instagram.com/prajakta_official?igshid=OGQ5ZDc2ODk2ZA==">Prajakta Mali Instagram</a>
