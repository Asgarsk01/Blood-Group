"use client"

import { useState, useEffect, useRef } from "react"
import { useRouter } from "next/navigation"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { toast } from "sonner"
import { ArrowLeft, Download, Mail, Loader2 } from "lucide-react"
import { saveData, getData, STORAGE_KEYS } from "@/lib/utils/storage"

interface Patient {
  id: string
  name: string
  age: number
  gender: string
  email: string
  collectionDate: string
  height: number
  weight: number
  bloodPressure: string
  medicalHistory?: string
  detectionHistory: string
}

export default function ReportPreview() {
  const router = useRouter()
  const [patient, setPatient] = useState<Patient | null>(null)
  const [detectionResult, setDetectionResult] = useState<string | null>(null)
  const [currentDate] = useState(new Date().toLocaleDateString())
  const [currentTime] = useState(new Date().toLocaleTimeString())
  const [isLoading, setIsLoading] = useState(true)
  const [isSending, setIsSending] = useState(false)
  const reportRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    // Get patient data and detection result from storage
    const patientData = getData<string | null>(STORAGE_KEYS.SELECTED_PATIENT, null);
    const result = getData<string | null>(STORAGE_KEYS.DETECTION_RESULT, null);

    console.log("Patient data from storage:", patientData)
    console.log("Detection result from storage:", result)

    if (patientData) {
      try {
        const parsedPatient = typeof patientData === 'string' ? JSON.parse(patientData) : patientData;
        setPatient(parsedPatient)

        // If we have a patient but no result, use the patient's detection history
        if (!result && parsedPatient.detectionHistory && parsedPatient.detectionHistory !== "Not Tested") {
          setDetectionResult(parsedPatient.detectionHistory)
        } else if (result) {
          setDetectionResult(result)

          // Update patient's detection history if needed
          if (parsedPatient.detectionHistory !== result) {
            const updatedPatient = {
              ...parsedPatient,
              detectionHistory: result,
            }
            setPatient(updatedPatient)

            // Update in storage
            saveData(STORAGE_KEYS.SELECTED_PATIENT, updatedPatient);

            // Update in patients list
            updatePatientInList(updatedPatient)
          }
        }
      } catch (error) {
        console.error("Error parsing patient data:", error)
        toast.error("Could not load patient data");
      }
    } else if (result) {
      // If we have a result but no patient, we can still show the result
      setDetectionResult(result)
    }

    setIsLoading(false)
  }, [])

  // Helper function to update patient in the patients list
  const updatePatientInList = (updatedPatient: Patient) => {
    try {
      const patients = getData<Patient[]>(STORAGE_KEYS.PATIENTS_LIST, []);
      const updatedPatients = patients.map((patient) => 
        patient.id === updatedPatient.id ? updatedPatient : patient
      );
      saveData(STORAGE_KEYS.PATIENTS_LIST, updatedPatients);
      console.log("Updated patient list in storage", updatedPatients);
    } catch (error) {
      console.error("Error updating patient in list:", error);
    }
  }

  // const handleDownloadPDF = () => {
  //   if (!patient) {
  //     toast.error("Cannot generate PDF without patient data")
  //     return
  //   }

  //   // In a real application, you would use a library like jsPDF or html2pdf
  //   // For this demo, we'll simulate the PDF generation

  //   toast.success("Please wait while we prepare your report");

  //   // Simulate PDF generation delay
  //   setTimeout(() => {
  //     toast.success( `Report saved as ${patient.id}_Report.pdf`);
  //   }, 1500)
  // }

  const handleDownloadPDF = async () => {
    if (!patient || !reportRef.current) {
      toast.error("Cannot generate PDF without patient data");
      return;
    }
  
    toast.success("Please wait while we prepare your report");
  
    try {
      const html2canvas = (await import("html2canvas-pro")).default;
      const jsPDF = (await import("jspdf")).default;
  
      const canvas = await html2canvas(reportRef.current);
      const imgData = canvas.toDataURL("image/png");
  
      const pdf = new jsPDF("p", "mm", "a4");
      const imgWidth = 210; // A4 width in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
  
      pdf.addImage(imgData, "PNG", 0, 0, imgWidth, imgHeight);
      pdf.save(`${patient.id}_Report.pdf`);
  
      toast.success(`Report saved as ${patient.id}_Report.pdf`);
    } catch (error) {
      console.error("Error generating PDF:", error);
      toast.error("Failed to generate PDF");
    }
  }  

  const handleSendEmail = async () => {
    if (!patient || !reportRef.current) {
      toast.error("Cannot send email without patient data");
      return;
    }

    setIsSending(true);
    toast.success(
      <div className="flex flex-col">
        <span className="font-medium">Preparing report for email</span>
        <span className="text-xs mt-1">Generating PDF and sending to {patient.email}</span>
      </div>,
      {
        duration: 3000,
      }
    );
    
    try {
      // Generate PDF
      const html2canvas = (await import("html2canvas-pro")).default;
      const jsPDF = (await import("jspdf")).default;
  
      const canvas = await html2canvas(reportRef.current);
      const imgData = canvas.toDataURL("image/png");
  
      const pdf = new jsPDF("p", "mm", "a4");
      const imgWidth = 210; // A4 width in mm
      const imgHeight = (canvas.height * imgWidth) / canvas.width;
  
      pdf.addImage(imgData, "PNG", 0, 0, imgWidth, imgHeight);
      
      // Convert PDF to base64 for sending via email
      const pdfBase64 = pdf.output('datauristring');
      
      // Send email with PDF attachment
      const response = await fetch('/api/send-report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: patient.email,
          patientName: patient.name,
          reportData: {
            id: patient.id,
            pdfBase64
          },
          reportDate: `${currentDate} ${currentTime}`,
          bloodGroup: detectionResult
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to send report email');
      }

      toast.success(
        <div className="flex flex-col">
          <span className="font-medium">Report sent successfully!</span>
          <span className="text-xs mt-1">Blood group report has been emailed to {patient.email}</span>
        </div>,
        {
          duration: 5000,
          icon: <Mail className="h-5 w-5 text-green-500" />,
        }
      );
    } catch (error) {
      console.error("Error sending email:", error);
      toast.error(
        <div className="flex flex-col">
          <span className="font-medium">Failed to send report</span>
          <span className="text-xs mt-1">Please try again or download the report instead</span>
        </div>,
        {
          duration: 5000,
        }
      );
    } finally {
      setIsSending(false);
    }
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p className="text-lg">Loading report data...</p>
        </div>
      </div>
    )
  }

  if (!detectionResult) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-lg font-medium">No detection result available</p>
          <Button onClick={() => router.push("/blood-detection")} className="mt-4">
            Return to Blood Detection
          </Button>
        </div>
      </div>
    )
  }

  // If we have a result but no patient, create a minimal report
  const showMinimalReport = !patient && detectionResult
  console.log(showMinimalReport);

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Report Preview</h1>
          <Button variant="outline" onClick={() => router.push("/blood-detection")} className="flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="flex justify-end gap-4 mb-6">
          <Button variant="outline" onClick={handleDownloadPDF} className="flex items-center gap-2" disabled={!patient}>
            <Download className="h-4 w-4" />
            Download PDF
          </Button>
          <Button 
            onClick={handleSendEmail} 
            className="flex items-center gap-2" 
            disabled={!patient || isSending}
          >
            {isSending ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin" />
                Sending...
              </>
            ) : (
              <>
                <Mail className="h-4 w-4" />
                Send to Email
              </>
            )}
          </Button>
        </div>

        <Card className="max-w-4xl mx-auto">
          <CardContent className="p-8" ref={reportRef}>
            <div className="flex justify-between items-center mb-8 border-b pb-4">
              <div className="flex items-center gap-4">
                <div className="w-16 h-16 bg-gray-200 rounded-full flex items-center justify-center">
                  <Image src="/placeholder.svg?height=64&width=64" alt="Logo" width={40} height={40} />
                </div>
                <div>
                  <h2 className="text-2xl font-bold">Blood Group Detection Report</h2>
                  <p className="text-muted-foreground">Sri Siddhartha Institute</p>
                </div>
              </div>
              <div className="text-right text-sm text-muted-foreground">
                <p>Report Date: {currentDate}</p>
                <p>Report Time: {currentTime}</p>
              </div>
            </div>

            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4">Patient Information</h3>
              {patient ? (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Patient ID</p>
                    <p className="font-medium">{patient.id}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Full Name</p>
                    <p>{patient.name}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Age</p>
                    <p>{patient.age} years</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Gender</p>
                    <p>{patient.gender}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Email</p>
                    <p>{patient.email}</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Sample Collection Date</p>
                    <p>{patient.collectionDate}</p>
                  </div>
                </div>
              ) : (
                <div className="bg-muted p-4 rounded-md text-center">
                  <p className="text-muted-foreground">No patient data available</p>
                </div>
              )}
            </div>

            <div className="mb-8">
              <h3 className="text-lg font-semibold mb-4">Detection Results</h3>
              <div className="bg-gray-50 p-6 rounded-lg text-center">
                <p className="text-sm font-medium text-muted-foreground mb-2">Blood Group</p>
                <p className="text-5xl font-bold text-primary">{detectionResult}</p>
              </div>
            </div>

            {patient && (
              <div className="mb-8">
                <h3 className="text-lg font-semibold mb-4">Additional Information</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Height</p>
                    <p>{patient.height} cm</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Weight</p>
                    <p>{patient.weight} kg</p>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-muted-foreground">Blood Pressure</p>
                    <p>{patient.bloodPressure}</p>
                  </div>
                  {patient.medicalHistory && (
                    <div className="md:col-span-2">
                      <p className="text-sm font-medium text-muted-foreground">Medical History</p>
                      <p>{patient.medicalHistory}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="text-center text-sm text-muted-foreground pt-4 border-t">
              <p>Generated by Blood Group Detection System</p>
              <p>This is an automated report. No signature required.</p>
            </div>
          </CardContent>
        </Card>
      </main>
    </div>
  )
}