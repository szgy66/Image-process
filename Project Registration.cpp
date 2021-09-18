#include<Eigen/Dense>
#include<opencv2/core.hpp>
#include<opencv2/highgui.hpp>
#include<iostream>
#include<math.h>
#include<opencv2/opencv.hpp>
#include<opencv2/core/eigen.hpp>
#include<cstring>



using namespace cv;
using namespace std; 
using namespace Eigen;

const int patch = 40;
const int alph = 4;
const int stride = 40;
const int lens = 6;
const float substrict = 0.4;

Mat Pointmatch(Mat img1, Mat img2);
Mat select_roi(int m, int n, Mat img1, Mat lr_roi);

int main()
{
	Mat img1 = imread("hr.png", IMREAD_GRAYSCALE);  // 高质量图像
	Mat img2 = imread("lr.png", IMREAD_GRAYSCALE);  // 低质量图像
	int H = img2.rows;  // 输出行	
	int W = img2.cols;  // 输出列
	Mat warped_img;
	warped_img = Pointmatch(img1, img2);
	Mat mask = Mat::zeros(img2.size(), CV_32FC1);
	Mat corr = Mat::zeros(img2.size(), CV_32FC1);

	return 0;
}




Mat Pointmatch(Mat img1,  Mat img2) // 特征点变换函数
{	
	Ptr<AKAZE> detector = AKAZE::create();
	vector<KeyPoint> keypoints_obj;
	vector<KeyPoint> keypoints_scene;
	Mat descriptor_obj, descriptor_scene;
	detector->detectAndCompute(img1, Mat(), keypoints_obj, descriptor_obj);
	detector->detectAndCompute(img2, Mat(), keypoints_scene, descriptor_scene);
	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
	vector<DMatch> matches;
	matcher.match(descriptor_obj, descriptor_scene, matches);
	Mat akazeMatchesImg;
	vector<DMatch> goodMatches;
	double minDist = 100000, maxDist = 0;
	// 找出关键点之间距离最大值和最小值
	for (int i = 0; i < descriptor_obj.rows; i++) {
		double dist = matches[i].distance;
		if (dist < minDist) {
			minDist = dist;
		}
		if (dist > maxDist) {
			maxDist = dist;
		}
	}
	// 保存匹配距离小于1.5*minDist的点对
	for (int i = 0; i < descriptor_obj.rows; i++) {
		double dist = matches[i].distance;
		if (dist < 3 * minDist) {
			goodMatches.push_back(matches[i]);
		}
	}
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (unsigned int i = 0; i < goodMatches.size(); i++)
	{
		obj.push_back(keypoints_obj[goodMatches[i].queryIdx].pt);
		scene.push_back(keypoints_scene[goodMatches[i].trainIdx].pt);
	}
	Mat H = findHomography(obj, scene, CV_RANSAC);
	Mat warped_img;
	warpPerspective(img1, warped_img, H, img2.size(), INTER_LINEAR, BORDER_CONSTANT);  // 返回HR变换后的图像warped_img
	return warped_img;
}


Mat select_roi(int m, int n, Mat img1, Mat lr_roi)
{
	Mat hr_roi = img1(Rect(n, m, patch, patch));
	float a[patch*patch];
	int cnt1 = 0;
	for (int i = 0; i < patch; i++)
	{
		for (int j = 0; j < patch; j++)
			a[cnt1] = hr_roi.at<float>(i, j);
			cnt1++;
	}

	float b[patch*patch];
	int cnt2 = 0;
	for (int i = 0; i < patch; i++)
	{
		for (int j = 0; j < patch; j++)
			b[cnt2] = lr_roi.at<float>(i,j);
		cnt2++;
	}
	Mat c = Mat::zeros(patch*patch, 2, CV_32FC1);
	for (int i = 0; i < patch*patch; i++) {
		c.at<float>(i, 0) = a[i];
		c.at<float>(i, 1) = b[i];
	}
	int row = c.rows;
	int col = c.cols;
	MatrixXd Y(row, col), X(2, 2);
	cv2eigen(c, Y);
	X = Y.adjoint()*Y;
	X = X.array() / (Y.rows() - 1);  // Xij
	
	MatrixXd A = Y(Rect(0, 0, 1, patch*patch)), P(1);
	P = A*A.adjoint();
	P = P.array() / (A.rows() - 1);  // Xaa
	float Caa = sqrt(int(P));
	MatrixXd B = Y(Rect(1, 0, 1, patch*patch)), Q(1, 1);
	Q = B.adjoint()*B;
	Q = Q.array() / (B.rows() - 1);  // Xbb
	Q = sqrt(Q);
	

}
