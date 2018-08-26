import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Enumeration;
import java.util.List;

public class Writer {
    private List<String> linesToWrite = new ArrayList<>();
    public void write(String file, List<String> lines) {
        try {
            Files.write(Paths.get(file), lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void write(Path file, List<String> lines) {
        try {
            Files.write(file, lines, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void writeFold(Path file, Instances instances){
        List<String> lines = new ArrayList<>();
        lines.add("@relation " + instances.relationName());
        for(int i = 0; i < instances.numAttributes()-1; i++){
            Attribute a = instances.attribute(i);
            String type = null;
            switch(a.type()) {
                case 0:
                    type = "NUMERIC";
                    break;
                case 1:
                    type = "NOMINAL";
                    break;
                case 2:
                    type = "STRING";
                    break;
                case 3:
                    type = "DATE";
                    break;
                case 4:
                    type = "RELATIONAL";
                    break;
            }
            lines.add("@attribute " + a.name() + " " + type);
        }
        Enumeration<Object> classValues= instances.classAttribute().enumerateValues();
        String classAttr = "{" + classValues.nextElement();
        while(classValues.hasMoreElements()){
            classAttr += "," + classValues.nextElement();
        }
        lines.add("@attribute " + "Class " + classAttr + "}");
        lines.add("@data");
        for(Instance i : instances){
            lines.add(i.toString());
        }
        write(file, lines);
    }
    public void writeFold(String file,Instances instances){
        List<String> lines = new ArrayList<>();
        lines.add("@relation " + instances.relationName());
        for(int i = 0; i < instances.numAttributes()-1; i++){
            Attribute a = instances.attribute(i);
            String type = null;
            switch(a.type()) {
                case 0:
                    type = "NUMERIC";
                    break;
                case 1:
                    type = "NOMINAL";
                    break;
                case 2:
                    type = "STRING";
                    break;
                case 3:
                    type = "DATE";
                    break;
                case 4:
                    type = "RELATIONAL";
                    break;
            }
            lines.add("@attribute " + a.name() + " " + type);
        }
        Enumeration<Object> classValues= instances.classAttribute().enumerateValues();
        String classAttr = "{" + classValues.nextElement();
        while(classValues.hasMoreElements()){
            classAttr += "," + classValues.nextElement();
        }
        lines.add("@attribute " + "Class " + classAttr + "}");
        lines.add("@data");
        for(Instance i : instances){
            lines.add(i.toString());
        }
        write(file, lines);
    }
    public void write(String file) {
        try {
            Files.write(Paths.get(file), linesToWrite, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void write(Path file) {
        try {
            Files.write(file, linesToWrite, Charset.forName("UTF-8"));
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    public void add(String line){
        linesToWrite.add(line);
    }
}
