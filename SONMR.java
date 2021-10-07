import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.*;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.hash.Hash;


public class SONMR {

    // first round map phase
    public static class Mapper1 extends Mapper<Object, Text, Text, NullWritable>{
        private int dataset_size;
        private int transactions_per_block;
        private int min_supp;
        private double corr_factor;

        // finding candidates for frequent items
        // start by setting up local variables
        public void setup(Context context1) throws IOException{
            Configuration conf = context1.getConfiguration();
            dataset_size = conf.getInt("dataset_size", dataset_size);
            transactions_per_block = conf.getInt("transactions_per_block", transactions_per_block);
            // lower the support per formula provided in hw instructions
            min_supp = conf.getInt("min_supp", min_supp);
            corr_factor = conf.getDouble("corr_factor", corr_factor);
            min_supp = (int)(corr_factor * ((double) min_supp / dataset_size) * transactions_per_block);
            System.err.println("The minimum support is " + min_supp);
        }

        // will be used for the output value, since at this phase we don't care about the output value for each
        // candidate frequent itemset
        private final static NullWritable one = NullWritable.get();

        private Text intKey = new Text();

        // now we move to the map phase
        public void map(Object key, Text value, Context context1
        ) throws IOException, InterruptedException {
            // hashmaps of hashsets used to hold candidate itemsets and non-candidate itemsets
            HashMap<HashSet<Integer>, Integer> notCands = new HashMap<>();
            HashMap<HashSet<Integer>, Integer> nextCands = new HashMap<>();
            HashMap<HashSet<Integer>, Integer> currCands = new HashMap<>();
            // arraylist of hashsets used to hold transactions
            ArrayList<HashSet<Integer>> transHold = new ArrayList<>();

            // we start by finding itemsets of size 1, from which we will build itemsets of size 2, 3, etc.
            // so we need to first split the transaction(s) by " " in order to extract single integers

            // split the file into individual transactions by newline character
            for (String a : value.toString().split("\\r?\\n")){
                // create a hashset to store the integers in each transaction
                HashSet<Integer> f = new HashSet<>();
                //System.err.println(a);
                // split each transaction into individual elements
                for (String b : a.split("\\s+")){
                    // add the integer to the hashmap used to store the transaction
                    f.add(Integer.parseInt(b));
                    // now create a hashset for the individual integer we are looking at
                    HashSet<Integer> c = new HashSet<>();
                    c.add(Integer.parseInt(b));
                    // if the hashmap already contains the element
                    if (currCands.containsKey(c)){
                        // increment that element's support by 1
                        currCands.put(c, currCands.get(c)+1);
                    } else {
                        // otherwise, put the element into the hashmap with a support of 1
                        currCands.put(c, 1);
                    }
                }
                // add the transaction to the arraylist of hashsets
                transHold.add(f);
            }
            System.err.println(currCands.toString());

            HashMap<HashSet<Integer>, Integer> frequents = new HashMap<>();
            // now we actually check if each element is frequent or not by comparing its support to min threshold
            for (Map.Entry<HashSet<Integer>, Integer> entry : currCands.entrySet()) {
                // set entry we are looking at to next position in iterator
                // if the entry's value is greater than min_supp, keep in currCands and write to output
                if (entry.getValue() >= min_supp) {
                    frequents.put(entry.getKey(), entry.getValue());
                    System.err.println("this is frequent " + entry.getKey());
                    String outputString = "";
                    for (Integer n : entry.getKey()) {
                        outputString = n + " ";
                    }
                    intKey.set(outputString);
                    context1.write(intKey, one);
                }
                // else if the entry's value is less than min_supp, add to notCands and remove from currCands
                else {
                    System.err.println("this is not frequent " + entry.getKey());
                    notCands.put(entry.getKey(), entry.getValue());
                    //itr.remove();
                }
            }

            currCands = frequents;
            // refers to size of itemsets we are looking for
            int k = 1;

            while (!currCands.isEmpty()){
                Iterator<Map.Entry<HashSet<Integer>, Integer>> itr3 = currCands.entrySet().iterator();
                Iterator<Map.Entry<HashSet<Integer>, Integer>> itr4 = currCands.entrySet().iterator();
                // create a new candidate set by joining 2 HashSets in currCands together (using two for loops)
                while (itr3.hasNext()){
                    Map.Entry<HashSet<Integer>, Integer> entry1 = itr3.next();
                    while (itr4.hasNext()){
                        Map.Entry<HashSet<Integer>, Integer> entry2 = itr4.next();
                        // create a temporary set by merging together two hashsets
                        HashSet<Integer> tempSet = mergeSet(entry1.getKey(), entry2.getKey());
                        // check if tempSet is in notCands
                        if (notCands.containsKey(tempSet) || nextCands.containsKey(tempSet)){
                            continue;
                        }
                        // check size of tempSet - we are only interested in hashsets of size k+1
                        if (tempSet.size() == k+1){
                            // iterate over all integers in tempSet
                            for (Integer e : tempSet){
                                // create a new temporary hashset that is (tempSet - a random integer)
                                HashSet<Integer> copyTempset = new HashSet<>(tempSet);
                                copyTempset.remove(e);
                                // if a subset of size k is not frequent, then tempSet cannot be frequent
                                // so add tempSet to notCands
                                if (notCands.containsKey(copyTempset) || !currCands.containsKey(copyTempset)){
                                    notCands.put(tempSet, 1);
                                    break;
                                }
                            }
                            // if all subsets of size k are frequent, then tempSet could be frequent
                            // so add tempSet to nextCands
                            System.err.println("this is a candidate for a frequent itemset " + tempSet);
                            nextCands.put(tempSet, 0);
                        }
                    }
                }

                // now, we have a hashmap (nextCands) of hashsets that may be frequent itemsets
                // that we have filtered by checking that all subsets of such hashsets are frequent
                // so we look over the block of transactions and calculate the support of each
                for (HashSet<Integer> g : transHold){
                    // for each hashset in nextCands
                    for (Map.Entry<HashSet<Integer>, Integer> h : nextCands.entrySet()){
                        // if the transaction contains the candidate itemset, increment its support by 1
                        if (g.containsAll(h.getKey())){
                            nextCands.put(h.getKey(), h.getValue() +1);
                        }
                    }
                }

                // at this point we need to filter out all elements of nextCands that are less than min_supp
                Iterator<Map.Entry<HashSet<Integer>, Integer>> itr2 = nextCands.entrySet().iterator();
                while (itr2.hasNext()){
                    Map.Entry<HashSet<Integer>, Integer> entry2 = itr2.next();
                    // if the element of nextCands has a support greater than min_supp, write to reducer
                    if (entry2.getValue() >= min_supp){
                        // we go through all elements of entry2.getKey()
                        String outputString = "";
                        for (Integer m : entry2.getKey()){
                            outputString = outputString + m + " ";
                        }
                        intKey.set(outputString);
                        context1.write(intKey, one);
                    }
                    // if the element of nextCands has a support less than min_supp, remove
                    else {
                        notCands.put(entry2.getKey(), entry2.getValue());
                        itr2.remove();
                    }
                }

                currCands = nextCands;
                nextCands = new HashMap<>();
                notCands = new HashMap<>();
                k++;
                //System.err.println("reached the bottom of big while loop");
            }
            //System.err.println("reached the end of the big while loop!");
        }


        // helper method for merging two hashsets together
        public static HashSet<Integer> mergeSet(HashSet<Integer> a, HashSet<Integer> b){
            HashSet<Integer> mergeSet = new HashSet<>();
            mergeSet.addAll(a);
            mergeSet.addAll(b);
            return mergeSet;
        }
    }

    // first round reduce phase
    // output all candidates from first round map phase
    public static class Reducer1 extends Reducer<Text, NullWritable, Text, NullWritable>{
        private final NullWritable result = NullWritable.get();
        public void reduce(Text key, Iterable<NullWritable> n, Context context1) throws IOException, InterruptedException {
            context1.write(key, result);
            //System.err.println("reached the reducer1 stage");
        }
    }

    // second round map phase
    public static class Mapper2 extends Mapper<Object, Text, Text, IntWritable>{
        private int dataset_size;
        private int transactions_per_block;
        private int min_supp;
        private double corr_factor;
        private ArrayList<HashSet<Integer>> holdCands = new ArrayList<>();
        private Text intKey2 = new Text();

        private final static IntWritable one = new IntWritable(1);

        public void setup (Context context) throws IOException{
            //System.err.println("reached the setup of mapper2 phase");
            Configuration conf = context.getConfiguration();
            dataset_size = conf.getInt("datset_size", dataset_size);
            transactions_per_block = conf.getInt("transactions_per_block", transactions_per_block);
            min_supp = conf.getInt("min_supp", min_supp);
            corr_factor = conf.getDouble("corr_factor", corr_factor);

            // for processing files in distributed cache
            URI[] cacheFiles = context.getCacheFiles();

            // read the file in the distributed cache that contains candidate frequent itemsets
            BufferedReader readSet =
                    new BufferedReader(new InputStreamReader(new FileInputStream(cacheFiles[0].toString())));

            // will likely need to change this - actually fixed it never mind
            // the issue I had here was trying to convert read output from mapper1 that were
            // hashsets.toString() which gave me brackets and commas
            for (String sw = readSet.readLine(); sw != null; sw = readSet.readLine()) {
                //System.err.println("reached the checking stage of mapper2 phase");
                // create hashset for each line of the file
                HashSet<Integer> hsh= new HashSet<>();
                // note each line of the file corresponds to a different candidate frequent itemset
                String[] arr = sw.split("\\s+");
                for (String l : arr){
                    // check if l is not a noninteger character
                    hsh.add(Integer.parseInt(l));
                }
                if (!holdCands.contains(hsh)){
                    holdCands.add(hsh);
                }
            }
            //System.err.println("Size of holdCands is " + holdCands.size());
        }

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            // we will attempt the second round of map/reduce processing one line of the transaction
            // at a time using the default InputFormatClass
            // we create a HashSet<Integer> to hold the integers in the transaction
            HashSet<Integer> tmp = new HashSet<>();
            // so we split each string of the transaction by " " to examine individual integers
            for (String j : value.toString().split("\\s")){
                // for each individual integer we find, we want to add it to a HashSet<Integer>
                tmp.add(Integer.parseInt(j));
            }
            //System.err.println(tmp);
            // after we have added all integers in the transaction to our HashSet<Integer> temp,
            // we want to see if tmp contains any of our candidate frequent itemsets
            // for each candidate in our set of candidate frequent itemsets
            for (HashSet<Integer> k : holdCands){
                // if the transaction we are looking at contains a candidate
                if (tmp.containsAll(k)){
                    // write to output a key-value pair for each candidate that appears in the transaction
                    String outputString = "";
                    for (Integer n : k){
                        outputString = outputString + n + " ";
                    }
                    intKey2.set(outputString);
                    context.write(intKey2, one);
                    //System.err.println(intKey2);
                }
            }
        }
    }

    // second round reduce phase
    public static class Reducer2 extends Reducer<Text, IntWritable, Text, IntWritable>{
        private int min_supp;
        // final output for final result
        private final static IntWritable result = new IntWritable();

        public void setup(Context context) throws IOException {
            Configuration conf = context.getConfiguration();
            min_supp = conf.getInt("min_supp", min_supp);
        }

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            // now we want to sum up all of the values associated with each key
            int sum = 0;
            for (IntWritable val : values) {
                sum += val.get();
            }
            result.set(sum);
            //System.err.println("Sum of the support is " + sum);
            //System.err.println("The min_supp is" + min_supp);
            // if the sum we have calculated is greater than or equal to the minimum support threshold
            // then we have found a frequent itemset! yay!
            if (sum >= min_supp){
                //System.err.println("This itemset is frequent!");
                context.write(key, result);
            }
        }
    }

    public static void main(String[] args) throws Exception{
        // number of transactions (positive integer)
        int dataset_size = Integer.parseInt(args[0]);
        // number of transactions that each call to map method in the first round will get (positive integer)
        int transactions_per_block = Integer.parseInt(args[1]);
        // minimium support threshold (positive integer)
        int min_supp = Integer.parseInt(args[2]);
        // "correction" factor to the minimum support threshold to be used in the first round (double between 0 and 1)
        double corr_factor = Double.parseDouble(args[3]);
        // path to the directory containing the input files
        String input_path = args[4];
        // path to the non-existing directory for the intermediate output (i.e. output of the Reducers in the first round)
        String interm_path = args[5];
        // path to the non-existing directory for the final output (i.e. the output of the Reducers in the second round)
        String output_path = args[6];

        // establishes configuration
        Configuration conf = new Configuration();
        // configures local variables to be used in mapper1 in job1
        conf.setInt("dataset_size", dataset_size);
        conf.setInt("transactions_per_block", transactions_per_block);
        conf.setInt("min_supp", min_supp);
        conf.setDouble("corr_factor", corr_factor);
        // creates job1 as an instance of job class using configuration specified
        Job job1 = Job.getInstance(conf, "job 1");
        org.apache.hadoop.mapreduce.lib.input.NLineInputFormat.setNumLinesPerSplit(job1,transactions_per_block);
        job1.setInputFormatClass(MultiLineInputFormat.class);
        job1.setMapperClass(Mapper1.class);
        job1.setCombinerClass(Reducer1.class);
        job1.setReducerClass(Reducer1.class);
        job1.setMapOutputKeyClass(Text.class);
        job1.setMapOutputValueClass(NullWritable.class);
        job1.setOutputKeyClass(Text.class);
        job1.setOutputValueClass(NullWritable.class);
        FileInputFormat.addInputPath(job1, new Path(input_path));
        FileOutputFormat.setOutputPath(job1, new Path(interm_path));
        if (job1.waitForCompletion(true)){
            Configuration conf2 = new Configuration();
            conf2.setInt("dataset_size", dataset_size);
            conf2.setInt("transactions_per_block", transactions_per_block);
            conf2.setInt("min_supp", min_supp);
            conf2.setDouble("corr_factor", corr_factor);
            Job job2 = Job.getInstance(conf2, "job 2");
            //job2.setInputFormatClass(MultiLineInputFormat.class);
            job2.setMapperClass(Mapper2.class);
            //job2.setCombinerClass(Reducer2.class);
            job2.setReducerClass(Reducer2.class);
            job2.setMapOutputKeyClass(Text.class);
            job2.setMapOutputValueClass(IntWritable.class);
            job2.setOutputKeyClass(Text.class);
            job2.setOutputValueClass(IntWritable.class);
            FileInputFormat.addInputPath(job2, new Path(input_path));
            FileOutputFormat.setOutputPath(job2, new Path(output_path));
            // adding intermediate output
            Path intermPath = new Path("/Users/maggiedrew/IdeaProjects/SONMR/interm1/part-r-00000");
            job2.addCacheFile(intermPath.toUri());
            System.exit(job2.waitForCompletion(true) ? 0 : 1);
        }
    }
}
