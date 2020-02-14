package neural_networkjava;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import org.apache.commons.math3.linear.RealMatrix;
import static org.apache.commons.math3.linear.MatrixUtils.createRealMatrix;

public class NeuralNetwork {
    private final int inputnodes, hiddennodes, outputnodes;
    private final float learningrate;
    private RealMatrix wih, who;
    private double activation_function;
    
    public NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, float learningrate){
        double[][] aux1;
        
        this.inputnodes = inputnodes;
        this.hiddennodes = hiddennodes;
        this.outputnodes = outputnodes;
        this.learningrate = learningrate;
        
        aux1 = gerarAleatório(hiddennodes, inputnodes);
        wih = createRealMatrix(aux1);
        aux1 = gerarAleatório(outputnodes, hiddennodes);
        who = createRealMatrix(aux1);       
    }
    
    public void train(List inputs_list, List targets_list){
        double[][] inputs = toArray2dinputs(inputs_list);
        double[][] targets = toArray2dtargets(targets_list);
        
        RealMatrix inputsM = createRealMatrix(inputs).transpose();
        RealMatrix targetsM = createRealMatrix(targets).transpose();
        
        RealMatrix hidden_inputs = wih.multiply(inputsM);
        RealMatrix hidden_outputs = activationFunction(hidden_inputs);
        
        RealMatrix final_inputs = who.multiply(hidden_outputs);
        RealMatrix final_outputs = activationFunction(final_inputs);
        
        RealMatrix output_errors = targetsM.subtract(final_outputs);
        
        RealMatrix hidden_errors = who.transpose().multiply(output_errors);
           
        
        RealMatrix num1 = output_errors.multiply(final_outputs);
        double[][] aux = new double[final_outputs.getRowDimension()][final_outputs.getColumnDimension()];
        for(double[] atual : aux){
            Arrays.fill(atual,1.0);
        }
        RealMatrix num2 = createRealMatrix(aux);        
        RealMatrix num3 = num2.subtract(final_outputs);
        num2 = num1.multiply(num3);
        
        who.add(num2.multiply(hidden_outputs.transpose()).scalarMultiply(learningrate));
        
        num1 = hidden_errors.multiply(hidden_outputs);
        
        aux = new double[hidden_outputs.getRowDimension()][hidden_outputs.getColumnDimension()];
        for(double[] atual : aux){
            Arrays.fill(atual,1.0);
        }
        num2 = createRealMatrix(aux);
        num3 = num2.subtract(hidden_outputs);
        num2 = num1.multiply(num3);
        
        wih.add(num2.multiply(inputsM.transpose()).scalarMultiply(learningrate));             
    }
    
    public RealMatrix query(List inputs_list){
        double[][] inputs = toArray2dinputs(inputs_list);       
        
        RealMatrix inputsM = createRealMatrix(inputs).transpose();        
        
        RealMatrix hidden_inputs = wih.multiply(inputsM);
        RealMatrix hidden_outputs = activationFunction(hidden_inputs);
        
        RealMatrix final_inputs = who.multiply(hidden_outputs);
        RealMatrix final_outputs = activationFunction(final_inputs);    
        
        return final_outputs;
    }
    
    private RealMatrix activationFunction(RealMatrix matriz){
        int i, j, rows = matriz.getRowDimension(), columns = matriz.getColumnDimension();
        double[][] result = new double[rows][columns];
        
        for(i = 0; i < rows; i++){
            for(j = 0; j < columns; j++){
                result[i][j] = sigmoidFunction(matriz.getEntry(i,j));
            }
        }
        
        return createRealMatrix(result); 
    }
    
    private double sigmoidFunction(double x) {
        return (1/( 1 + Math.pow(Math.E,(-1*x))));
    }
    
    private double[][] toArray2dtargets(List targets_list){
        int j;
        double[][] inputs = new double[1][10];
       for(j = 0; j < 10; j++){
            inputs[0][j] = (double) targets_list.get(j);            
        }           
        
        return inputs;
    }
    
    private double[][] toArray2dinputs(List inputs_list){
        int j;
        double[][] inputs = new double[1][784];
        for(j = 0; j < 784 ; j++){
                inputs[0][j] = (double) inputs_list.get(j);               
        }  
        return inputs;
    }
    
    private double[][] gerarAleatório(int rows, int columns){       
        double[][] retorno = new double[rows][columns];       
        
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < columns; j++){
                retorno[i][j] = ThreadLocalRandom.current().nextDouble(0.01, 1)-0.5;
            }
        }
        return retorno;
    } 
}
