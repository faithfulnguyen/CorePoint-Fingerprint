/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package corepoint;

import java.io.File;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.Arrays;
import org.bytedeco.javacpp.indexer.DoubleRawIndexer;
import org.bytedeco.javacpp.indexer.FloatRawIndexer;
import org.bytedeco.javacpp.indexer.UByteRawIndexer;
import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_core.BORDER_DEFAULT;
import static org.bytedeco.javacpp.opencv_core.CV_32F;
import static org.bytedeco.javacpp.opencv_core.CV_PI;
import org.bytedeco.javacpp.opencv_core.Mat;
import org.bytedeco.javacpp.opencv_core.Size;
import static org.bytedeco.javacpp.opencv_core.print;
import org.bytedeco.javacpp.opencv_imgcodecs;
import static org.bytedeco.javacpp.opencv_imgcodecs.imread;
import static org.bytedeco.javacpp.opencv_imgcodecs.imwrite;
import org.bytedeco.javacpp.opencv_imgproc;
import static org.bytedeco.javacpp.opencv_imgproc.CV_GRAY2RGB;
import static org.bytedeco.javacpp.opencv_imgproc.GaussianBlur;
import static org.bytedeco.javacpp.opencv_imgproc.resize;

/**
 *
 * @author nguyentrungtin
 */
public class CorePoint {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        File folder = new File("");
        String fileName = folder.getAbsolutePath() + "/src/finger/";
        File[] listOfFiles = new File(fileName).listFiles();
        Arrays.sort(listOfFiles);
        for(int i = 0; i < listOfFiles.length; i++){
            if (listOfFiles[i].getName().contains(".tif")){
                opencv_core.Mat img = imread (fileName + "/"+ listOfFiles[i].getName(), opencv_imgcodecs.CV_LOAD_IMAGE_GRAYSCALE);
                resize(img, img, new Size(120,120));
                Mat normailizedImg = normalizeSubWindow(img);
                CorePoint cr = new CorePoint();
                Mat[] ori = cr.localSmooth(normailizedImg);
                Mat smooth = cr.smoothedOrientation(ori[0], ori[1]);
                cr.poinCare(smooth, img, listOfFiles[i].getName());
            }
        }
      
            
    }
   
    public static opencv_core.Mat normalizeSubWindow(opencv_core.Mat image){
        int ut = 139;
        int vt = 100;//1607;  
        double u = meanMatrix(image);
        double v = variance(image, u);
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols() ; j++){
                double beta = Math.sqrt((vt * 1.0 / v ) * (Math.pow(idx.get(i, j) - u, 2)));
                if(idx.get(i, j) > ut){
                    idx.put(i, j, 0, (int)ut + (int)beta);
                }
                else idx.put(i, j, 0, Math.abs((int)ut - (int)beta));      
            }
        }
        return image;
    }
   
    public static double variance(opencv_core.Mat image, double mean){
        double var = 0; 
        UByteRawIndexer idx = image.createIndexer();
        for(int i = 0; i < image.rows(); i++){
            for(int j = 0; j < image.cols(); j++){
                var += Math.pow((idx.get(i, j) - mean), 2);
            }
        }
        var /= (image.cols() * image.rows());
        return var;
    }
    
    public static double meanMatrix(opencv_core.Mat img){
        double sum = 0;
        UByteRawIndexer idx = img.createIndexer();
        for(int i = 0; i < img.rows(); i++){
            for(int j = 0; j < img.cols(); j++){
                sum += idx.get(i, j, 0);
            }
        }
        sum /= (img.cols() * img.rows());
        return sum;  
    }
    
    public Mat[] sobelDerivatives(Mat img){
        opencv_core.MatExpr zero = Mat.zeros(new Size(img.rows(), img.cols()), CV_32F);
    	Mat grad_x = zero.asMat();
        Mat grad_y = zero.asMat();
        FloatRawIndexer xId = grad_x.createIndexer();
        FloatRawIndexer yId = grad_y.createIndexer();
        for(int i = 1; i < img.rows() - 1; i++){
            for(int j = 1; j < img.cols() - 1; j++){
                float px = sobelDerivativeX(j, i, img);
                float py = sobelDerivativeY(j, i, img);
                xId.put(i, j,  px);
                yId.put(i, j,  py);
            }
        }
        return new Mat[]{grad_x, grad_y};
    }
    
    public float sobelDerivativeX(int x, int y, Mat img){
        UByteRawIndexer idx = img.createIndexer();
        return -idx.get(x - 1, y - 1) - 2 * idx.get(x, y - 1) - idx.get(x + 1, y -1)
                + idx.get(x - 1, y + 1) + 2 * idx.get(x, y + 1) + idx.get(x + 1, y + 1);
                
    }
   
    public float sobelDerivativeY(int x, int y, Mat img){
        UByteRawIndexer idx = img.createIndexer();
        return -idx.get(x - 1, y - 1) - 2*idx.get(x - 1, y) - idx.get(x - 1, y + 1) 
                + idx.get(x + 1, y - 1) + 2*idx.get(x + 1, y) + idx.get(x + 1, y + 1);
    }
    
    public double blockOrientation( Mat blck_sbl_x, Mat blck_sbl_y){
    	double v_x = 0;
    	double v_y = 0;
        int w = blck_sbl_x.rows();
    	FloatRawIndexer idx_x = blck_sbl_x.createIndexer();
    	FloatRawIndexer idx_y = blck_sbl_y.createIndexer();
    	for(int u = 0; u < w; u++){
            for(int v = 0; v < w; v++){
                v_x += 2 * idx_x.get(u, v) * idx_y.get(u, v);
                v_y += (idx_x.get(u, v)*idx_x.get(u, v) - idx_y.get(u, v)*idx_y.get(u, v));
            }
    	}
    	double theta = 0.5 * ( Math.PI + Math.atan2(v_x, v_y));
    	return theta;
    }

    public Mat localOrientation(Mat img, Mat blck_sbl_x, Mat blck_sbl_y){
    	int w = 10;
    	Mat oriImg = new Mat(blck_sbl_x.rows()/ w, blck_sbl_x.cols()/ w, opencv_core.CV_64F);
    	DoubleRawIndexer idx = oriImg.createIndexer();
    	int ori_i = 0;
    	int ori_j = 0;
        int r =  blck_sbl_x.rows();
        int c= blck_sbl_y.cols();
    	for(int i = 1; i <  blck_sbl_x.rows(); i += w){
            for(int j = 1; j < blck_sbl_x.cols(); j += w){
                Mat sbX = blck_sbl_x.apply(new opencv_core.Rect(j, i, Math.min(w, r - j), Math.min(w, c - i)));
                Mat sbY = blck_sbl_y.apply(new opencv_core.Rect(j, i, Math.min(w, r - j), Math.min(w, c - i)));
                double ori = blockOrientation(sbX, sbY);
                idx.put(ori_i,ori_j, ori);
                ori_j ++;
            }
            ori_j = 0;
            ori_i ++;
        }
    	return oriImg;
    }
    
    public Mat[] localSmooth(Mat img){
    	Mat[] sb = sobelDerivatives(img);
    	Mat ori = localOrientation(img, sb[0], sb[1]);
    	Mat sinY = new Mat(ori.size(), ori.type());
    	Mat cosX = new Mat(ori.size(), ori.type());
    	DoubleRawIndexer idx_o = ori.createIndexer();
    	DoubleRawIndexer idx_x = cosX.createIndexer();
    	DoubleRawIndexer idx_y = sinY.createIndexer();
    	for(int i = 0; i < ori.rows(); i++){
            for(int j = 0; j < ori.cols(); j++){
                idx_x.put(i, j, Math.cos(2 * idx_o.get(i, j)));
                idx_y.put(i, j, Math.sin(2 * idx_o.get(i, j)));
            }
    	}
    	return new Mat[]{ cosX, sinY};
    }

    public Mat smoothedOrientation(Mat cosX, Mat sinY){
        GaussianBlur(cosX, cosX, new Size(3, 3), 0, 0, BORDER_DEFAULT);
        GaussianBlur(sinY, sinY, new Size(3, 3), 0, 0, BORDER_DEFAULT);
        Mat smooth = cosX.clone();
        DoubleRawIndexer idx_s = smooth.createIndexer();
        DoubleRawIndexer idx_x =  cosX.createIndexer();
        DoubleRawIndexer idx_y = sinY.createIndexer();
        for(int i = 0; i < smooth.rows(); i++){
            for(int j = 0; j < smooth.cols(); j++){
                double theta = 0.5 * Math.atan2(idx_y.get(i, j), idx_x.get(i,j));
                idx_s.put(i, j, theta);
            }
        }
    	return smooth;
    }
    public void poinCare(Mat smooth, Mat img, String name){
        Mat siglar = smooth.clone();
    	DoubleRawIndexer index = siglar.createIndexer();
        Mat rgb = new Mat();
        opencv_imgproc.cvtColor(img, rgb, CV_GRAY2RGB );
        UByteRawIndexer idxI = rgb.createIndexer();
        int w = 3;
        int i =  smooth.rows();
        int j =  smooth.cols();
    	for(int r = 1; r < smooth.rows() - 1; r ++){
            for(int c = 1; c < smooth.cols() - 1; c ++){
                double beta = calcNeighbors(smooth, r, c);
                index.put(r, c, beta);
            }
    	}
        for(int r = 0; r < siglar.rows(); r++){
            for(int c = 0; c < siglar.cols(); c++){
                if(index.get(r, c) >= 0.50 && index.get(r, c) <= 0.51){
                    for(int k = 0; k < 10; k++){
                        idxI.put(c*10 + k, r*10 + k, 0, 150);
                        idxI.put(c*10 + k, r*10 + k, 1, 0);
                        idxI.put(c*10 + k, r*10 + k, 2, 0);
                    }
                    
                }
            }
        }
        imwrite(name, rgb);
    }
 
    public double calcNeighbors(Mat smooth, int i, int j){
        DoubleRawIndexer idx = smooth.createIndexer();
        double beta = 0;
        double pc = Math.abs((idx.get(i + 1, j - 1)) - (idx.get(i + 1, j))); // O2 - O1
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i, j - 1)) - (idx.get(i + 1, j - 1))); // O3 - O2
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i - 1, j - 1)) - (idx.get(i, j - 1))); //04- O3
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i - 1, j)) - (idx.get(i - 1, j - 1))); // 05 - 04
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i - 1, j + 1)) - (idx.get(i - 1, j))); // 06 - 05
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i, j + 1)) - (idx.get(i - 1, j + 1))); // 07 - 06
        beta += checkConditional(pc);
        
        pc = Math.abs((idx.get(i + 1, j + 1)) - (idx.get(i, j + 1))); // 08 - 07
        beta += checkConditional(pc);
       
        beta /= ( 2 * CV_PI);
        return beta;   
        
    }
    
    public double checkConditional(double pc){
        double beta = 0;
        if(Math.abs(pc) < CV_PI / 2)
            beta = pc;
        else if(pc < -CV_PI /2)
            beta = CV_PI + pc;
        else beta = CV_PI - pc;
        return beta;
    }

}
