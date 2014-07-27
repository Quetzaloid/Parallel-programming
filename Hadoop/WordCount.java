package org.myorg; 

import java.io.IOException; 
import java.util.*; 

import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.conf.*; 
import org.apache.hadoop.io.*; 
import org.apache.hadoop.mapred.*; 
import org.apache.hadoop.util.*; 

public class WordCount {

      public static class Map extends MapReduceBase implements Mapper<LongWritable, Text, Text, Text> { 
          private Text file = new Text(); 
          private Text word = new Text(); 
          private String pattern = "[^a-zA-Z]";               

          public void map(LongWritable key, Text value, OutputCollector<Text, Text> output, Reporter reporter) throws IOException { 
            FileSplit fileSplit = (FileSplit) reporter.getInputSplit();
            String filename = fileSplit.getPath().getName();
            file.set(filename);

            String line = value.toString(); 
            line = line.replaceAll(pattern, " ");
            StringTokenizer tokenizer = new StringTokenizer(line); 
            while (tokenizer.hasMoreTokens()) { 
              word.set(tokenizer.nextToken()); 
              output.collect(word, file); 
           } 
         } 
      } 

      public static class Reduce extends MapReduceBase implements Reducer<Text, Text, Text, IntWritable> { 
         @Override
         public void reduce(Text key, Iterator<Text> values, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException { 
         int sum = 0;
         String tWord = key.toString();
         Set tCount = new HashSet(); 
         while (values.hasNext()) {
           Text victim = values.next();
           String tFile = victim.toString();
           if (tCount.add(new Count(tWord,tFile))) {
                sum +=1;
           }
         } 
         output.collect(key, new IntWritable(sum)); 
       } 
     } 
      static class Count{
         String word;
         String file;

         public Count(String word, String file) {
             this.word = word;
             this.file = file;
         }

         public int hashCode() {
             return 1;
         }

         public boolean equals(Object obj) {
             Count ct=(Count) obj;
             return this.word==ct.word && this.file.equals(ct.file);
         }
     }
      public static void main(String[] args) throws Exception { 
        JobConf conf = new JobConf(WordCount.class); 
        conf.setJobName("WordCount"); 

        FileSystem fs = FileSystem.get(conf);         
        fs.delete(new Path(args[1]),true);
         
        conf.setOutputKeyClass(Text.class); 
        conf.setOutputValueClass(IntWritable.class); 

        conf.setMapOutputKeyClass(Text.class);
        conf.setMapOutputValueClass(Text.class);

        conf.setMapperClass(Map.class); 
        //conf.setCombinerClass(Map.class); 
        conf.setReducerClass(Reduce.class); 

        conf.setInputFormat(TextInputFormat.class); 
        conf.setOutputFormat(TextOutputFormat.class); 

        FileInputFormat.setInputPaths(conf, new Path(args[0])); 
        FileOutputFormat.setOutputPath(conf, new Path(args[1])); 

        JobClient.runJob(conf); 
    } 
}