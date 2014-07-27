package org.myorg; 

import java.io.IOException;
import java.io.*;
import java.net.URI; 
import java.util.*; 

import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.conf.*; 
import org.apache.hadoop.io.*; 
import org.apache.hadoop.mapred.*; 
import org.apache.hadoop.util.*; 
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;

public class Jacobi extends Configured implements Tool {
      public static final int    max_iter = 100;
      public static final double eps      = 1e-10;

      public static boolean stopIteration(Configuration conf) throws IOException{
          FileSystem fs=FileSystem.get(conf);
          Path preFile=new Path("preX/Result");
          Path curFile=new Path("curX/part-00000");

          if(!(fs.exists(preFile) && fs.exists(curFile))){
              System.exit(1);
          }

          boolean stop = true;
          String line1,line2;
          FSDataInputStream in1 = fs.open(preFile);
          FSDataInputStream in2 = fs.open(curFile);
          InputStreamReader isr1 = new InputStreamReader(in1);
          InputStreamReader isr2 = new InputStreamReader(in2);
          BufferedReader br1 = new BufferedReader(isr1);
          BufferedReader br2 = new BufferedReader(isr2);

          while((line1=br1.readLine())!=null && (line2=br2.readLine())!=null){
              String []str1=line1.split("\\s+");
              String []str2=line2.split("\\s+");
              double preElem = Double.parseDouble(str1[1]);
              double curElem = Double.parseDouble(str2[1]);
              if(Math.abs(preElem - curElem) > eps){
                  stop=false;
                  break;
              }
          }

          if(stop==false){
              fs.delete(preFile, true);
              if(fs.rename(curFile, preFile)==false){
                 System.exit(1);
              }
          }
          return stop;
      }


      public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, DoubleWritable> { 
         private  double[]    sumVec;
         private  double[]    resVec;
         private  double[]    diaVec;
         int nsize;
         public void configure(JobConf conf) {
           try{
               Path       vInput;
               FileSystem fs;
               URI[]      fvector;
               nsize = conf.getInt("DIMENTION", 0);
               sumVec = new double[nsize];
               resVec = new double[nsize];
               diaVec = new double[nsize];
               Arrays.fill(sumVec, 0);
               Arrays.fill(resVec, 0);
               Arrays.fill(diaVec, 0);

               fvector = DistributedCache.getCacheFiles(conf);
               vInput = new Path(fvector[0].getPath());
               fs = FileSystem.get(URI.create("hdfs://node17.cs.rochester.edu:9000"), conf);

               FSDataInputStream fdis = fs.open(vInput);

               String line; 
               while((line=fdis.readLine()) != null){
                 StringTokenizer tokenizer = new StringTokenizer(line);
                 int    rowIdx = Integer.parseInt(tokenizer.nextToken());
                 int    colIdx = Integer.parseInt(tokenizer.nextToken());           
                 double matVar = Double.parseDouble(tokenizer.nextToken());               
                 if (rowIdx==colIdx){
                    diaVec[rowIdx] = matVar;
                 }
                 else if (colIdx==nsize){
                    resVec[rowIdx] = matVar;
                 }
                 else{
                    sumVec[rowIdx] += matVar;
                 }
               }
           }catch(IOException e){
               e.printStackTrace();
           }
        }
         public void map(LongWritable key, Text value, OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException {
           String line = value.toString(); 
           StringTokenizer tokenizer = new StringTokenizer(line);

           int    rowIdx=0;
           double xValue=0;

           if (tokenizer.hasMoreTokens()) {
              rowIdx = Integer.parseInt(tokenizer.nextToken());
              xValue = Double.parseDouble(tokenizer.nextToken());
           }

           double xResult = (resVec[rowIdx]-(sumVec[rowIdx]*xValue))/diaVec[rowIdx];
           output.collect (new IntWritable(rowIdx), new DoubleWritable(xResult));
        }
      } 

      public static class Reduce extends MapReduceBase implements Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> { 
         @Override
         public void reduce(IntWritable key, Iterator<DoubleWritable> values, OutputCollector<IntWritable, DoubleWritable> output, Reporter reporter) throws IOException { 
         output.collect(key, values.next()); 
       }
     } 
      @Override
      public int run(String[] args) throws Exception {
         Configuration conf = getConf();
         FileSystem fs = FileSystem.get(conf);
         JobConf job = new JobConf(conf);
         job.setJarByClass(Jacobi.class);
         
         fs.delete(new Path("curX"),true);
         job.setInputFormat(TextInputFormat.class);
         job.setOutputFormat(TextOutputFormat.class);
         job.setMapperClass(Map.class);
         job.setReducerClass(Reduce.class);
         job.setOutputKeyClass(IntWritable.class);
         job.setOutputValueClass(DoubleWritable.class);

         FileInputFormat.setInputPaths(job, new Path("preX"));
         FileOutputFormat.setOutputPath(job, new Path("curX"));

         JobClient.runJob(job);
         return 1;
     }
      public static void main(String[] args) throws Exception { 
        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        
         Path matFile = new Path(args[0]);
         FSDataInputStream matData = fs.open(matFile);
         BufferedReader br = new BufferedReader(new InputStreamReader(matData));
                                                                     
         int i = 0;
         String line;
         while ((line = br.readLine()) != null) {
            StringTokenizer tokenizer = new StringTokenizer(line);
            String  iRow = tokenizer.nextToken();
            String  iCol = tokenizer.nextToken();
            if (Integer.parseInt(iRow)==Integer.parseInt(iCol)){
               i++;
            }
         } 
         br.close();
         int dimention = i;

         conf.setInt("DIMENTION", dimention);
         Path xFile = new Path("preX/Result");
         FSDataOutputStream xData = fs.create(xFile);
         BufferedWriter iniX = new BufferedWriter(new OutputStreamWriter(xData));
         for(int j=0; j<dimention; j++){
            iniX.write(String.valueOf(j) + " 0");
            iniX.newLine();
         }
         iniX.close();

         URI matVec = new URI(args[0]);
         DistributedCache.addCacheFile(matVec, conf);

         int iteration = 0;
         do {
               ToolRunner.run(conf, new Jacobi(), args);
         } while (iteration++ < max_iter && (!stopIteration(conf)));
    } 
}
