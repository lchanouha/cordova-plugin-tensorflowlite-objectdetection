#import "TensorFlowLite.h"
#import <Cordova/CDVPlugin.h>
#import "TFLTensorFlowLite.h"

@interface TensorFlowLite ()

@property(nonatomic) TFLInterpreter *interpreter;

@end

@implementation TensorFlowLite

- (void)loadModel:(CDVInvokedUrlCommand*)command
{
    CDVPluginResult* pluginResult = nil;
    NSString* number = [command.arguments objectAtIndex:0];
    NSString* modelPath =  [command.arguments objectAtIndex:1];
    
    
    TFLInterpreterOptions *options = [[TFLInterpreterOptions alloc] init];
    options.numberOfThreads = 2;
    NSError *error;
    self.interpreter = [[TFLInterpreter alloc] initWithModelPath:modelPath
                                                    options:options
                                                      error:&error];

    if (self.interpreter == nil || error != nil) {
        NSString *results =
        [NSString stringWithFormat:@"Failed to create the interpreter due to error:%@",
         error.localizedDescription];
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_ERROR messageAsString:results];
    } else {
        pluginResult = [CDVPluginResult resultWithStatus:CDVCommandStatus_OK messageAsString:@"Model loaded"];
    }

    [self.commandDelegate sendPluginResult:pluginResult callbackId:command.callbackId];
}

@end
