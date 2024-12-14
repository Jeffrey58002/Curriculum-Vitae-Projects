
final int clusterAllAmount = 100;

final int drawSpeed = 60; // 60
//float[] brushSizes = {0.5, 0.7, 1.0, 0.6, 2.0};
//float[] brushSizes = {0.3, 0.6, 0.8, 1.0, 1.6}; //07_19
float[] brushSizes = {0.05, 0.06, 0.08, 0.1, 0.2}; 
PImage[] brushTypes;
String srcImgName = "./cluster/resize.jpg";
PImage srcImg;
PImage bg;
PImage logo;
int widthUnit = (1920/3)+1;
int heightUnit = (1080/5) +1;

JSONObject labelAll;
JSONObject labelEyes;
JSONObject labelIndex;
JSONArray currentClusterArray;
String clusterName;
JSONObject cluster;
JSONArray clusterIndex;
int currentClusterNo = 0;
int cluster_size = 0;
int targetIndex2Draw = 0;
int pixelIndex = 0;
int eyesIndex = 0;
int x = 0;
int y = 0;
int routateNo = 0;
int rnd = 0;
int pixelsPerIteration = 700 ;
void setup() {
  size(1920, 1080);
  colorMode(HSB,100); // !
  fullScreen();
  setImages();
  bg = loadImage("./image_s/bg.png"); //loadImage("./image_s/bg.png");
  // logo = loadImage("./image_s/aia.jpg");
  image(bg, 0, 0, 1920, 1080);
  // image(logo,0,1720,1080,200);
  labelAll = loadJSONObject("./cluster/cluster_all.json");
  labelEyes = loadJSONObject("./cluster/cluster_eyes.json");
  labelIndex = loadJSONObject("./cluster/cluster_index.json");
  clusterIndex = getClusterIndexByName("index");
  println("Current Cluster No:"+currentClusterNo);
  frameRate(drawSpeed);
  /*
  drawSpeed 30: 91s
  drawSpeed 45: 61s
  drawSpeed 60: 46s
  */
}


void draw() {
  if(currentClusterNo < clusterAllAmount){
    /*
    Draw All cluster
    */
    int currentClusterIndex = clusterIndex.getJSONArray(currentClusterNo).getInt(0);
    int currentClusterSize = clusterIndex.getJSONArray(currentClusterNo).getInt(1);
    clusterName = "cluster_"+currentClusterIndex;
    cluster = getClusterAllByName(clusterName);
    cluster_size = cluster.getInt("gloup_size");
    currentClusterArray = cluster.getJSONArray("cluster");
    
    //targetIndex2Draw = pixelIndex+50; //original +700
    
    // If you don't want it to go out of bound.
    int maxPixelsPerIteration = 1920 * 1080;
    int pixelsPerIteration = 300; // Example value    300 is decent
    pixelsPerIteration = min(pixelsPerIteration, maxPixelsPerIteration);
    
    
    //int pixelsPerIteration = 50;  // Number of pixels to paint in each iteration
    int totalPixels = currentClusterArray.size();
    
    
    if(targetIndex2Draw > currentClusterArray.size()){
      targetIndex2Draw = currentClusterArray.size();
    }
        for (int i = pixelIndex; i < min(pixelIndex + pixelsPerIteration, totalPixels); i++) {
      x = currentClusterArray.getJSONArray(i).getInt(0);
      y = currentClusterArray.getJSONArray(i).getInt(1);
      routateNo = int(y / heightUnit) * 3 + int(x / widthUnit);
      
      int bs = 1;
      int bt = int(random(5));
      
      //There is a set amount of pixels in a 1920*1080 background , so the higher the "clusterAllAmount" is set ,
      //the less pixels will be included in a single cluster , which means you will need to adjust the if statement below
      //alongside the "clusterAllAmount" variable , if "clusterAllAmount" is set too high but the if statement stays 
      //the same, then all the cluster size will be too small, causing the if statement to only run in "else" condition.
      
      if (currentClusterSize > 30000) {
        bs = 4;
      } else if (currentClusterSize > 20000) {
        bs = 3;
      } else if (currentClusterSize > 15000) {
        bs = 2;
      } else if (currentClusterSize > 10000) {
        bs = 1;
      } else {
        bs = 0;
      }
      //int density = 100 * (bs + 1);
      
      //if (bs == 0) {                        Original
      //  density = 200;
      //} else if (bs == 1) {
      //  density = 300;
      //}
      int density = 25 * (bs + 1);
      
      if (bs == 0) {
        density = 100;
      } else if (bs == 1) {
        density = 150;
      }
      
      /*
      if (routateNo == 4 || routateNo == 7) {
        bs = 0;
        density = 300;
      } else if (routateNo == 1 || routateNo == 3 || routateNo == 5 || routateNo == 6 || routateNo == 8) {
        bs = 2;
        density = 100 * (bs + 1);
      }*/
      
      rnd = int(random(density));
      if (rnd < 1) {      //original: 1 (1% chance of drawing)
        paintPixel(x, y, bt, bs, routateNo, 0);
      }
    }
    
    pixelIndex += pixelsPerIteration;  // Update the pixelIndex
    
    if (pixelIndex >= totalPixels) {
      pixelIndex = 0;
      currentClusterNo++;
      println("Current Cluster No:" + currentClusterNo);
    }
  } else {
    println("Process time = " + int(millis() / 1000));
    String fileName = "./ai_works/ai_work_" + month() + "." + day() + "-" + hour() + "." + minute() + "." + second() + ".png";
    saveFrame(fileName);
    delay(10000);
    exit();
  }
}


void  setImages() {
  srcImg = loadImage(srcImgName);
  srcImg.loadPixels();
  brushTypes = new PImage[]{loadImage("./selected/0.png"), loadImage("./selected/1.png"), loadImage("./selected/2.png"), loadImage("./selected/3.png"), loadImage("./selected/4.png"), loadImage("./selected/5.png")};
}

JSONObject getClusterAllByName(String clusterName) {
  JSONObject clusterGroup = labelAll.getJSONObject(clusterName);
  return clusterGroup;
}

JSONObject getClusterEyesByName(String clusterName) {
  JSONObject clusterGroup = labelEyes.getJSONObject(clusterName);
  return clusterGroup;
}
JSONArray getClusterIndexByName(String clusterName) {
  JSONArray clusterGroup = labelIndex.getJSONArray(clusterName);
  return clusterGroup;
}

void paintPixel(int tmpX, int tmpY, int brushTypeNo, int brushSizeNo, int direction, int r) {
  imageMode(CENTER);
  PImage tmpBrush = brushTypes[brushTypeNo];
  int loc = tmpX + tmpY*srcImg.width;
  color pixelColor = srcImg.pixels[loc];
  //pixelColor = color(hue(pixelColor), saturation(pixelColor)+random(75), brightness(pixelColor)+random(25) );
  pixelColor = color(0,0, brightness(pixelColor)); // -random(30)
  //float bright = brightness(pixelColor);
  fill(pixelColor);//
  tint(pixelColor, 65); // default = 65
  image(tmpBrush, tmpX, tmpY);
  pushMatrix();
  translate(tmpX,tmpY);
  rotate(random(2.3562, 2.8798));
  scale(brushSizes[brushSizeNo]);
  image(tmpBrush, 0, 0);
  popMatrix();
  
}


//int getTint(float brightness){
//  //If the brightness is too dark in cetain area , we set Tint to a lower value(to make it less dark) .  
//  int value = 0 ;
//  if(brightness < 25)
//  {
//    value = 30 ;
//  }
//  else{ value = 65 ; }

//  return value ;
//}



float getRotateAngle(int direction) {
  float value = 0;
  float offset = (0.16*PI);
  //float dice  = int(random(1)) ;
  //if(dice == 0)
  //{
  //  value = random((0.75 * PI)+offset, (1.25 * PI)-offset); //Default if there's no condition
  //}
  //else { value = random((PI)+offset, (1.5 * PI)-offset);} //  <===== the cross was drawn by this line
  value = random((0.75 * PI)+offset, (1.25 * PI)-offset);
  return value; 
}


/*
float getRotateAngle(int direction) {
  float value = 0;
  float offset = (0.16*PI);
  if (direction == 3 || direction == 6 || direction == 11 || direction == 14) {
    value = random((PI)+offset, (1.5 * PI)-offset); // 208.8 ~ 241.2
  } else if (direction == 4 || direction == 7 || direction == 10 || direction == 13) {
    value = random((1.25*PI)+offset, (1.75*PI)-offset);
  } else if (direction == 5 || direction == 8 || direction == 9 || direction == 12){
    value = random((0.25 * PI)+offset, (0.75 * PI)-offset);
  } else {
    value = random((0.75 * PI)+offset, (1.25 * PI)-offset);
  }

  return value; //random(-0.5, 0.3);
}*/
/*
 float getRotateAngle(int direction) {
   float value = 0;
   if (direction == 0 || direction == 3 || direction == 11 || direction == 14) {
    if(random(2) < 1){
       value = random(radians(30), radians(65));
     }else{
       value = random(radians(215), radians(245));
    }
   } else if (direction == 1 || direction == 4) {
     if(random(2) < 1){
       value = random(radians(15), radians(30));
     }else{
       value = random(radians(165), radians(195));
     }
   } else if (direction == 10 || direction == 13) {
     if(random(2) < 1){
       value = random(radians(75), radians(105));
     }else{
       value = random(radians(260), radians(295));
     }
   } else if (direction == 2 || direction == 5 || direction == 9 || direction == 12){
     if(random(2) < 1){
       value = random(radians(110), radians(160)); // random(radians(110), radians(160));
     }else{
       value = random(radians(290), radians(340));  // random(radians(290), radians(340));
     }
   } else {
     if(random(2) < 1){
       value = random(radians(0), radians(15));
     }else{
       value = random(radians(165), radians(195));
     }
   }

   return value; //random(-0.5, 0.3);
 }
 */
 /*
   sort by color
   cluster size
   brush size
   density
   rotate
   color
 */
