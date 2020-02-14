package neural_networkjava;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Scanner;


public class Teste {
    public static void main(String args[]){
        NeuralNetwork n = new NeuralNetwork(784,100,10, (float) 0.3);
        
        String filename = "C:\\Users\\Thiago\\Documents\\mnist_train_100.csv";        
        File arq = new File(filename);
        ArrayList<ArrayList<Double>> inputs = new ArrayList();
        ArrayList<ArrayList<Double>> targets = new ArrayList();
        ArrayList<Double> data = new ArrayList();
        ArrayList<Double> data2 = new ArrayList();         
        
        try{
            Scanner inputStream = new Scanner(arq);  
            String[][] separa = new String[100][785];
            while(inputStream.hasNext()){
                for(int i = 0;i < 100;i++){ 
                   for(int k = 0; k < 10; k++){
                        data2.add(0.0);
                   }       
                   separa[i] = inputStream.next().split(",");
                   data2.set((int) Double.parseDouble(separa[i][0]),0.99);
                   targets.add(data2);
                   for(int j = 1; j < 785; j++){                       
                       data.add((Double.parseDouble(separa[i][j])/255.0*0.99)+0.01);                                            
                   }
                   inputs.add(data);
                   data2 = new ArrayList();
                }                
            }
            inputStream.close();
        }catch(FileNotFoundException e){
            e.printStackTrace();
        } 
        
        
        
        for(int i = 0; i < inputs.size(); i++){            
            n.train(inputs.get(i), targets.get(i));
        }
        
        filename = "C:\\Users\\Thiago\\Documents\\mnist_test_10.csv";        
        arq = new File(filename);
        inputs = new ArrayList();
        targets = new ArrayList();
        data = new ArrayList();
        data2 = new ArrayList();         
        
        try{
            Scanner inputStream = new Scanner(arq);  
            String[][] separa = new String[100][785];
            while(inputStream.hasNext()){
                for(int i = 0;i < 10;i++){ 
                   for(int k = 0; k < 10; k++){
                        data2.add(0.0);
                   }       
                   separa[i] = inputStream.next().split(",");
                   data2.set((int) Double.parseDouble(separa[i][0]),0.99);
                   targets.add(data2);
                   for(int j = 1; j < 785; j++){                       
                       data.add((Double.parseDouble(separa[i][j])/255.0*0.99)+0.01);                                            
                   }
                   inputs.add(data);
                   data2 = new ArrayList();
                }                
            }
            inputStream.close();
        }catch(FileNotFoundException e){
            e.printStackTrace();
        }
    }
}
