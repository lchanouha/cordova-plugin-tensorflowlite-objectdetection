#import <Cordova/CDVPlugin.h>

@interface TensorFlowLite : CDVPlugin

- (void)echo:(CDVInvokedUrlCommand*)command;

@end