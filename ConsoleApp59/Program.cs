using System.Diagnostics;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

Console.WriteLine("WARNING: This program attempts a 128-bit GUID search.");
Console.WriteLine("This will run for a practically infinite amount of time even millions of years in super computer.");
Console.WriteLine("But what I want to try it.");
Guid targetGuid = new Guid("00000000-0000-0000-0000-000000000001");
//Guid targetGuid = new Guid("00000000-0000-0000-0000-00000000000F"); // For trillion years
//Guid targetGuid = new Guid("00000000-0000-0000-0001-000000000000"); // For one year don't know what i'm saying
Console.WriteLine($"JOKE: WARNING: Your pc going to explode after million years after it finds '{targetGuid}'.");
Console.WriteLine("Press Ctrl+C to stop the experiment.");

int guidsPerThread = 1024;

Console.WriteLine($"Target GUID: {targetGuid}");

byte[] targetGuidBytes = targetGuid.ToByteArray();

using var context = Context.CreateDefault();
using var accelerator = context.CreateCudaAccelerator(0);
Console.WriteLine($"Using GPU: {accelerator.Name}");

var targetBufferGpu = accelerator.Allocate1D<byte>(16);
targetBufferGpu.CopyFromCPU(targetGuidBytes);

var resultFoundFlagGpu = accelerator.Allocate1D<int>(1);
var resultGuidLowGpu = accelerator.Allocate1D<ulong>(1);
var resultGuidHighGpu = accelerator.Allocate1D<ulong>(1);

static void GuidSearchKernel(
    Index1D globalThreadIndex,
    ArrayView<byte> targetGuidBytes,
    ArrayView<int> resultFoundFlag,
    ArrayView<ulong> resultGuidLow,
    ArrayView<ulong> resultGuidHigh,
    ulong baseGuidLow,
    ulong baseGuidHigh,
    int numGuidsPerThread)
{
    if (resultFoundFlag[0] != 0)
        return;

    ulong threadBlockStartOffsetLow = (ulong)globalThreadIndex * (ulong)numGuidsPerThread;

    ulong currentLow = baseGuidLow;
    ulong currentHigh = baseGuidHigh;

    if (currentLow > ulong.MaxValue - threadBlockStartOffsetLow)
    {
        currentHigh++;
    }
    currentLow += threadBlockStartOffsetLow;

    for (int i = 0; i < numGuidsPerThread; ++i)
    {
        if (resultFoundFlag[0] != 0)
            return;

        bool match = true;
        for (int k = 0; k < 8; ++k)
        {
            if (((byte)(currentLow >> (k * 8))) != targetGuidBytes[k])
            {
                match = false;
                break;
            }
        }
        if (match)
        {
            for (int k = 0; k < 8; ++k)
            {
                if (((byte)(currentHigh >> (k * 8))) != targetGuidBytes[k + 8])
                {
                    match = false;
                    break;
                }
            }
        }

        if (match)
        {
            if (Atomic.CompareExchange(ref resultFoundFlag[0], 0, 1) == 0)
            {
                resultGuidLow[0] = currentLow;
                resultGuidHigh[0] = currentHigh;
            }
            return;
        }

        if (currentLow == ulong.MaxValue)
        {
            currentLow = 0;
            currentHigh++;
        }
        else
        {
            currentLow++;
        }
    }
}

var kernel = accelerator.LoadAutoGroupedStreamKernel<
    Index1D,
    ArrayView<byte>,
    ArrayView<int>,
    ArrayView<ulong>,
    ArrayView<ulong>,
    ulong,
    ulong,
    int>(GuidSearchKernel);

long desiredThreadsPerLaunch = 1024L * 1024L * 4L;
long deviceMaxThreads = (long)accelerator.MaxNumThreadsPerGroup * accelerator.MaxGridSize.X;
long threadsToConsider = Math.Min(desiredThreadsPerLaunch, deviceMaxThreads);
long practicalCap = 1024L * 1024L * 16L;
if (threadsToConsider > practicalCap)
{
    threadsToConsider = practicalCap;
}
if (threadsToConsider <= 0)
{
    threadsToConsider = 1024L * 256L;
}
if (threadsToConsider > int.MaxValue)
{
    threadsToConsider = int.MaxValue;
}
int numThreadsPerLaunch = (int)threadsToConsider;

Console.WriteLine($"Launching kernel with {numThreadsPerLaunch} threads per call, {guidsPerThread} GUIDs per thread.");
ulong guidsProcessedPerKernelCall = (ulong)numThreadsPerLaunch * (ulong)guidsPerThread;
Console.WriteLine($"Total GUIDs per kernel launch: {guidsProcessedPerKernelCall:N0}");

ulong currentBaseLow = 0;
ulong currentBaseHigh = 0;
bool found = false;
long totalScannedSinceLastLog = 0;
long grandTotalScanned = 0;
long logInterval = Math.Max(10_000_000_000L, (long)guidsProcessedPerKernelCall * 5);

Stopwatch sw = Stopwatch.StartNew();
Stopwatch logSw = Stopwatch.StartNew();

while (!found)
{
    resultFoundFlagGpu.MemSetToZero();
    resultGuidLowGpu.MemSetToZero();
    resultGuidHighGpu.MemSetToZero();

    kernel(numThreadsPerLaunch,
           targetBufferGpu.View,
           resultFoundFlagGpu.View,
           resultGuidLowGpu.View,
           resultGuidHighGpu.View,
           currentBaseLow,
           currentBaseHigh,
           guidsPerThread);
    accelerator.Synchronize();

    int[] hostFlag = resultFoundFlagGpu.GetAsArray1D();
    if (hostFlag[0] == 1)
    {
        found = true;
        ulong[] hostLow = resultGuidLowGpu.GetAsArray1D();
        ulong[] hostHigh = resultGuidHighGpu.GetAsArray1D();

        byte[] foundBytes = new byte[16];
        for (int i = 0; i < 8; ++i) foundBytes[i] = (byte)(hostLow[0] >> (i * 8));
        for (int i = 0; i < 8; ++i) foundBytes[i + 8] = (byte)(hostHigh[0] >> (i * 8));

        Guid finalFoundGuid = new Guid(foundBytes);
        Console.WriteLine($"\n🎉🎉🎉 Found GUID: {finalFoundGuid} 🎉🎉🎉");
        Console.WriteLine($"Matched target: {finalFoundGuid == targetGuid}");
        break;
    }

    grandTotalScanned += (long)guidsProcessedPerKernelCall;
    totalScannedSinceLastLog += (long)guidsProcessedPerKernelCall;

    if (totalScannedSinceLastLog >= logInterval)
    {
        double elapsedSecondsForLog = logSw.Elapsed.TotalSeconds;
        if (elapsedSecondsForLog > 0.001)
        {
            double guidsPerSecond = totalScannedSinceLastLog / elapsedSecondsForLog;
            Console.WriteLine($"Scanned {grandTotalScanned:N0} total. Rate: {guidsPerSecond:N0} GUIDs/sec. Next High:Low {currentBaseHigh:X16}:{currentBaseLow:X16}");
        }
        else
        {
            Console.WriteLine($"Scanned {grandTotalScanned:N0} total. (Log interval too short for rate). Next High:Low {currentBaseHigh:X16}:{currentBaseLow:X16}");
        }
        totalScannedSinceLastLog = 0;
        logSw.Restart();
    }

    ulong oldLow = currentBaseLow;
    currentBaseLow += guidsProcessedPerKernelCall;
    if (currentBaseLow < oldLow && guidsProcessedPerKernelCall > 0)
    {
        currentBaseHigh++;
        if (currentBaseHigh == 0 && oldLow > currentBaseLow)
        {
            Console.WriteLine("\nScanned all 2^128 possibilities (this is practically impossible).");
            found = true;
        }
    }
}

sw.Stop();
logSw.Stop();
Console.WriteLine($"\nExperiment finished or stopped.");
Console.WriteLine($"Total GUIDs scanned in this session: {grandTotalScanned:N0}");
Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F2} seconds.");
if (sw.Elapsed.TotalSeconds > 0)
{
    Console.WriteLine($"Average rate: {grandTotalScanned / sw.Elapsed.TotalSeconds:N0} GUIDs/sec.");
}


if (!found)
{
    Console.WriteLine("Target GUID was not found.");
}