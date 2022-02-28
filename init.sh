if [ -z ${TVM_HOME+x} ]; then 
    echo "TVM_HOME is unset"; 
else 
    echo "TVM_HOM is set to '$TVM_HOME'";
    version=$(python3 -c 'import tvm; print(tvm.__version__)')
    if [ "$version" = "0.8.0" ]; then
        echo "version" $version "compatible!"  
        cmd="cp res/tvm/* ${TVM_HOME}"
        echo $cmd
    else
        echo "version" $version "is not compatible. Currently, only 0.8.0 is available"
    fi 
fi
