package org.myorg; 

import java.io.IOException;
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

public class MatVecMul {

      public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, IntWritable, FloatWritable> {       
         private Path vInput;
         private FileSystem fs;
         private URI[] fvector;
         ArrayList<String>  vector;

         public void configure(JobConf conf) {
           try{
               fvector = DistributedCache.getCacheFiles(conf);
               vInput = new Path(fvector[0].getPath());
               fs = FileSystem.get(conf);
               vector = new ArrayList<String>();
               FSDataInputStream fdis = fs.open(vInput);
               String temp;
               while((temp=fdis.readLine()) != null)
                  {
                    vector.add(temp);
                  }
           }catch(IOException e){
               e.printStackTrace();
           }
        }

         public void map(LongWritable key, Text value, OutputCollector<IntWritable, FloatWritable> output, Reporter reporter) throws IOException { 
           
           String line = value.toString(); 
           StringTokenizer tokenizer = new StringTokenizer(line);

           if (tokenizer.hasMoreTokens()) {
             int   rowIdx = Integer.parseInt(tokenizer.nextToken());
             int   colIdx = Integer.parseInt(tokenizer.nextToken());           
             float matVar = Float.parseFloat(tokenizer.nextToken());

             float vecVar = Float.parseFloat(vector.get(colIdx-1);
             output.collect (new IntWritable(rowIdx), new FloatWritable(matVar*vecVar));
           }
        }
     }

      public static class Reduce extends MapReduceBase implements Reducer<IntWritable, FloatWritable, IntWritable, FloatWritable> { 
         @Override
         public void reduce(IntWritable key, Iterator<FloatWritable> values, OutputCollector<IntWritable, FloatWritable> output, Reporter reporter) throws IOException { 
         float sum = 0;
         while (values.hasNext()) {
           sum += values.next().get();
         } 
         output.collect(key, new FloatWritable(sum)); 
       } 
     } 

      public static void main(String[] args) throws Exception { 
         JobConf conf = new JobConf(MatVecMul.class); 
         conf.setJobName("MatVecMul"); 

         FileSystem fs = FileSystem.get(conf);         
         fs.delete(new Path(args[1]),true);

         conf.setOutputKeyClass(IntWritable.class); 
         conf.setOutputValueClass(FloatWritable.class); 

         conf.setMapOutputKeyClass(IntWritable.class);
         conf.setMapOutputValueClass(FloatWritable.class);

         conf.setMapperClass(Map.class); 
         //conf.setCombinerClass(Map.class); 
         conf.setReducerClass(Reduce.class);

         conf.setInputFormat(TextInputFormat.class); 
         conf.setOutputFormat(TextOutputFormat.class); 

         URI vector = new URI("vector.dat");
         DistributedCache.addCacheFile(vector, conf);

         FileInputFormat.setInputPaths(conf, new Path(args[0])); 
         FileOutputFormat.setOutputPath(conf, new Path(args[1])); 

         JobClient.runJob(conf); 
    } 
}
