/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package corepoint;

/**
 *
 * @author nguyentrungtin
 */
public class Point{
    
    private float x;
    private float y;

    public Point() {
        x = 0;
        y = 0;
    }
    public Point(float x, float y) {
        this.x = x;
        this.y = y;
    }
    public float getX(){
        return x;
    }
    
    public float getY(){
        return y;
    }
    public void setX(float x){
        this.x = x;
    }
    public void setY(float y){
        this.y = y;
    }
    public void setValue(float x, float y){
        this.x = x;
        this.y = y;
    }
    public double distanceAnotherPoint(Point diff){
        return Math.sqrt((this.x - diff.x) * (this.x - diff.x) + (this.y - diff.y)*(this.y - diff.y) );
    }
    
}
