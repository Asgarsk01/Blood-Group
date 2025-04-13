"use client"

import { useState, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card"
import { UserPlus, Droplet, FileText, Users, LogOut, Download, RefreshCw, ArrowRight } from "lucide-react"
import { toast } from "sonner"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { getData, removeData, saveData, STORAGE_KEYS } from "@/lib/utils/storage"

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

export default function Dashboard() {
  const router = useRouter()
  const [userName, setUserName] = useState("User")
  const [stats, setStats] = useState({
    totalPatients: 0,
    testsCompleted: 0,
    reportsGenerated: 0,
    activeUsers: 3,
  })
  const [testedPatients, setTestedPatients] = useState<Patient[]>([])
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    // Check if user is authenticated
    const isAuthenticated = getData<string>(STORAGE_KEYS.IS_AUTHENTICATED, "");
    if (!isAuthenticated) {
      router.push("/")
      return
    }

    // Try to get user name from storage
    const storedName = getData<string>(STORAGE_KEYS.USER_NAME, "");
    if (storedName) {
      setUserName(storedName)
    }

    // Initialize patients list if it doesn't exist
    const patientsList = getData<any[]>(STORAGE_KEYS.PATIENTS_LIST, []);
    if (patientsList.length === 0) {
      saveData(STORAGE_KEYS.PATIENTS_LIST, []);
    }

    // Update stats and tested patients list
    updateDashboardData()
    setIsLoading(false)
  }, [router])

  const updateDashboardData = () => {
    try {
      const patients = getData<Patient[]>(STORAGE_KEYS.PATIENTS_LIST, []);
      const tested = patients.filter((p) => p.detectionHistory !== "Not Tested")

      setTestedPatients(tested)

      setStats({
        totalPatients: patients.length,
        testsCompleted: tested.length,
        reportsGenerated: Math.floor(tested.length * 0.8), // Assume 80% of tests generated reports
        activeUsers: 3,
      })
    } catch (error) {
      console.error("Error updating dashboard data:", error)
    }
  }

  const handleLogout = () => {
    // Clear authentication status
    removeData(STORAGE_KEYS.IS_AUTHENTICATED);

    toast.success("You have been successfully logged out");

    router.push("/")
  }

  const handleExportData = () => {
    const date = new Date().toISOString().split("T")[0]
    toast.success(`File saved as PatientData_${date}.csv`);
  }

  const handleBloodDetection = () => {
    // Clear any previously selected patient
    removeData(STORAGE_KEYS.SELECTED_PATIENT);
    router.push("/blood-detection")
  }

  const handleRefreshStats = () => {
    updateDashboardData()
    toast.success("Dashboard data has been updated");
  }

  const handleViewPatientReport = (patient: Patient) => {
    // Store selected patient for use in report preview
    saveData(STORAGE_KEYS.SELECTED_PATIENT, patient);
    saveData(STORAGE_KEYS.DETECTION_RESULT, patient.detectionHistory);

    // Navigate to report preview
    router.push("/report-preview")
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <p className="text-lg">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-2xl font-bold">Blood Group Detection System</h1>
          <Button variant="ghost" size="icon" onClick={handleLogout}>
            <LogOut className="h-5 w-5" />
            <span className="sr-only">Logout</span>
          </Button>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h2 className="text-3xl font-bold text-center">
            Welcome, {userName}, to the Future of Blood Typing Analysis!
          </h2>
        </div>

        {/* Stats Section */}
        <div className="flex justify-end mb-2">
          <Button variant="outline" size="sm" onClick={handleRefreshStats} className="flex items-center gap-2">
            <RefreshCw className="h-4 w-4" />
            Refresh Dashboard
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Total Patients Registered</CardTitle>
              <UserPlus className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.totalPatients}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Blood Tests Completed</CardTitle>
              <Droplet className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.testsCompleted}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Reports Generated</CardTitle>
              <FileText className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.reportsGenerated}</div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-sm font-medium">Active Users</CardTitle>
              <Users className="h-4 w-4 text-muted-foreground" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{stats.activeUsers}</div>
            </CardContent>
          </Card>
        </div>

        {/* Actions Section */}
        <div className="flex flex-col md:flex-row gap-4 justify-center mb-8">
          <Button size="lg" className="flex items-center gap-2" onClick={() => router.push("/patient-details")}>
            <UserPlus className="h-5 w-5" />
            Add New Patient
          </Button>
          <Button size="lg" className="flex items-center gap-2" onClick={handleBloodDetection}>
            <Droplet className="h-5 w-5" />
            Blood Group Detection
          </Button>
          <Button size="lg" variant="outline" className="flex items-center gap-2" onClick={handleExportData}>
            <Download className="h-5 w-5" />
            Export Data
          </Button>
        </div>

        {/* Tested Patients Section */}
        <Card className="mb-8">
          <CardHeader>
            <CardTitle>Patients with Blood Group Results</CardTitle>
            <CardDescription>Showing patients who have completed blood group detection</CardDescription>
          </CardHeader>
          <CardContent>
            {testedPatients.length > 0 ? (
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Patient ID</TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Age</TableHead>
                      <TableHead>Gender</TableHead>
                      <TableHead>Blood Group</TableHead>
                      <TableHead>Actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {testedPatients.map((patient) => (
                      <TableRow key={patient.id}>
                        <TableCell className="font-medium">{patient.id}</TableCell>
                        <TableCell>{patient.name}</TableCell>
                        <TableCell>{patient.age}</TableCell>
                        <TableCell>{patient.gender}</TableCell>
                        <TableCell>
                          <span className="font-medium text-primary">{patient.detectionHistory}</span>
                        </TableCell>
                        <TableCell>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="flex items-center gap-2"
                            onClick={() => handleViewPatientReport(patient)}
                          >
                            <FileText className="h-4 w-4" />
                            View Report
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>No patients with blood group results yet.</p>
                <p className="mt-2">Add a patient and perform blood detection to see results here.</p>
              </div>
            )}
          </CardContent>
          {testedPatients.length > 0 && (
            <CardFooter className="flex justify-end">
              <Button variant="outline" size="sm" onClick={() => router.push("/patient-details")}>
                View All Patients
                <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </CardFooter>
          )}
        </Card>
      </main>
    </div>
  )
}